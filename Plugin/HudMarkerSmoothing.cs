using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;
using Sandbox.Game.Entities;
using Sandbox.Game.Gui;
using Sandbox.Game.GUI.HudViewers;
using Sandbox.Game.World;
using Sandbox.Graphics;
using Sandbox.Graphics.GUI;
using VRage.Generics;
using VRage.Utils;
using VRageMath;
using VRageRender;
using VRageRender.Messages;

namespace SmoothFrames
{
	/// <summary>
	///     HUD GPS / POI markers project against <c>MySector.MainCamera</c>
	///     (vanilla, sim-tick-baked) once per sim tick from the
	///     <c>MyGuiSandbox.Draw</c> chain. Marker icon and labels stay frozen
	///     at the sim-tick screen position across every render frame between
	///     sim ticks while the world geometry slides smoothly, reading as
	///     stutter against the smoothed scene.
	///
	///     Fix shape: tag every render message that carries the marker's
	///     screen position (icon atlas, text labels) at sim emission time
	///     with the originating world coord plus the original sim-tick
	///     screen position. Then per render frame, in
	///     <see cref="RenderFrameSmoothing"/>'s postfix on
	///     <c>ProcessMessageQueue</c>, walk the tracked-sprite dictionary
	///     and SET each message's position field to
	///     <c>original + (smoothed_screen − vanilla_screen)</c> recomputed
	///     against the smoothed camera at THIS render frame's timestamp.
	///     Setting (vs incrementing) prevents accumulation across the
	///     multiple render frames that re-read the same persisted message
	///     between sim ticks. The mutation lands BEFORE
	///     <c>RenderMainSprites</c> consumes the bucketed messages, so the
	///     downstream draw picks up the fresh per-frame position.
	///
	///     Earlier prefix-on-<c>ProcessDrawMessage</c> approach: that prefix
	///     only fires once per sim tick (sprite messages are only drained
	///     from the bucket on the first render frame after each sim
	///     emission), producing a fixed offset for the entire inter-tick —
	///     same per-tick jump magnitude as vanilla, just time-shifted by 1
	///     tick → indistinguishable on screen. Replaced by the
	///     per-render-frame loop below.
	///
	///     Tagged message types and position fields:
	///     - <see cref="MyRenderMessageDrawSpriteAtlas"/> — <c>Position</c>
	///       in HUD coords [0..1].
	///     - <see cref="MyRenderMessageDrawString"/> /
	///       <see cref="MyRenderMessageDrawStringAligned"/> —
	///       <c>ScreenCoord</c> in pixels (the engine converts the projected
	///       normalized coord via
	///       <see cref="MyGuiManager.GetScreenCoordinateFromNormalizedCoordinate(Vector2,bool)"/>
	///       at sim time, so we apply the same conversion to the delta
	///       before adding).
	///
	///     Marker text labels are emitted from
	///     <c>MyGuiScreenHudBase.DrawTexts</c>, which runs AFTER
	///     <c>MyHudMarkerRender.Draw</c> returns and the thread-static
	///     marker context is cleared. The context is bound to each
	///     <see cref="MyHudText"/> at <c>Start</c> time (see
	///     <see cref="PatchMyHudTextStart"/>) and re-installed by
	///     <see cref="PatchDrawTexts"/> immediately before each text's inner
	///     <c>DrawString</c> call.
	/// </summary>
	internal static class HudMarkerSmoothing
	{
		// Sim thread per-marker context. Set by TryComputeScreenPoint postfix
		// (success path), cleared by it on failure. Inherited by every
		// sprite-emit call that follows for the same marker until the next
		// TryComputeScreenPoint or the end of the marker draw.
		[ThreadStatic] private static MarkerContext _currentContext;

		// True only while MyHudMarkerRender.Draw is on the stack (set by
		// PatchMyHudMarkerRenderDraw). TryComputeScreenPoint is a general
		// world→screen helper the engine also calls outside Draw — e.g. to
		// project a grid target reticule — and the context it sets is only
		// cleared by Draw's postfix. Honoring an out-of-Draw call would leave a
		// marker context active when unrelated HUD (the toolbar) is drawn next,
		// tagging its sprites and Swapping them onto the smoothed marker
		// position. PatchTryComputeScreenPoint sets context only within this
		// scope.
		[ThreadStatic] internal static bool InMarkerDraw;

		// Sim thread populates from MyRenderProxy.EnqueueMessage postfix while
		// a marker context is set; render thread reads (and mutates) in the
		// per-render-frame refresh path. Entries are not removed on read —
		// the render thread re-uses them across the multiple render frames
		// that share the same sim emission. Three cleanup paths:
		//   1. Sim re-emission with marker context active overwrites the
		//      entry with a fresh ctx (see OnMessageEnqueued).
		//   2. The render thread drops the entry when the pooled message
		//      goes back to MyMessagePool (see OnMessagePoolReturn) — the
		//      race-closing path: by the time sim's next-tick MessagePool.Get
		//      hands the instance to a new consumer, no stale entry remains
		//      for render's refresh loop to latch onto and overwrite the new
		//      consumer's Position with the marker's smoothed Position.
		//   3. Render-frame TTL evicts anything not refreshed within
		//      _staleThresholdTicks, as a safety net for cases (1)/(2)
		//      somehow miss.
		private static readonly ConcurrentDictionary<MyRenderMessageBase, MarkerContext> _trackedSprites =
			new ConcurrentDictionary<MyRenderMessageBase, MarkerContext>();

		// Eviction threshold for tracked sprites — well above sim tick
		// interval (16.67 ms at 60 Hz) so a single dropped tick won't evict
		// a still-active marker. Anything not refreshed by sim emission
		// within this window has stopped being drawn.
		private static readonly long _staleThresholdTicks = Stopwatch.Frequency / 5; // 200 ms

		// Render-thread scratch for the per-frame eviction pass; reused to
		// avoid an allocation when there's nothing to evict.
		private static readonly List<MyRenderMessageBase> _evictionScratch =
			new List<MyRenderMessageBase>();

		// MyHudText → its associated marker context. MyHudText labels (the
		// distance/name text under each marker) are emitted via
		// MyGuiScreenHudBase.DrawTexts which runs AFTER
		// MyHudMarkerRender.Draw returns and our thread-static marker context
		// is cleared. We capture the marker context onto the MyHudText at
		// Start time (when context is still active), then PatchDrawTexts
		// re-installs it per iteration before the inner DrawString call.
		// Pooled MyHudText instances re-bind on each Start, so the table
		// doesn't grow unboundedly.
		private static readonly ConditionalWeakTable<MyHudText, MarkerContext> _hudTextContexts =
			new ConditionalWeakTable<MyHudText, MarkerContext>();

		internal sealed class MarkerContext
		{
			public Vector3D WorldPos;
			// Normalized HUD coords [0..1] — what TryComputeScreenPoint
			// returned at sim emission. Used as the reference point for the
			// per-render-frame delta (smoothedScreen - VanillaScreenPos).
			public Vector2 VanillaScreenPos;

			// Original message-position values captured at OnMessageEnqueued
			// time, before we mutated. Each render frame we SET the message's
			// position to (Original + delta) — never increment — so
			// successive render frames don't accumulate the delta.
			public Vector2 OriginalAtlasPosition;
			public Vector2 OriginalStringScreenCoord;
			public Vector2 OriginalSpriteRectPos;
			public MessageKind Kind;

			// True when MyHudMarkerRender.PointOfInterest.Draw selected the
			// off-screen-arrow branch for this emission — i.e., the marker
			// projected outside the UI-scaled safe-rect. The engine then
			// overwrites projectedPoint2D to a point on a 0.77-radius ellipse
			// around hud center pointing toward the marker, and emits the
			// DirectionIndicator atlas + text labels at that overwritten
			// position. Our standard `Position += delta_normalized × hudSize`
			// mutation produces the wrong motion for that path because the
			// engine's clamp is non-linear in the projection. The mutation
			// switches to recomputing the ellipse position from the
			// smoothed-camera projection when this is set.
			public bool IsOffScreenArrow;

			// Engine's ellipse-clamped HUD-pixel position at sim emission — our
			// replication of the off-screen-arrow overwrite that we can't see
			// directly because TryComputeScreenPoint returns before that
			// block runs. Used as the reference for the off-screen mutation's
			// shift (new_ellipse - OldEllipsePosition).
			public Vector2 OldEllipsePosition;

			// True when this context tags the piloting crosshair sprite (set by
			// PatchMyHudCrosshair, not by the marker render path). The
			// crosshair has a separate world-anchor formula
			// (`seat.GetPosition() + 1000 * seat.Forward`, fresh at sim T),
			// uses the piloted-ship's CURRENT-pose correction (not the
			// prev-pose POI variant), and stores its atlas Position in mixed
			// units (X normalized, Y in HUD-pixel space) — different from
			// the on-screen marker atlas (both axes in HUD-pixel space).
			// ApplyMutation branches on this flag to take the
			// crosshair-specific path.
			public bool IsCrosshair;

			// Stopwatch tick at which this context was last refreshed by
			// OnMessageEnqueued. Used by the render-thread refresh loop to
			// evict entries whose marker stopped emitting (engine retired the
			// pool slot without re-running our cleanup branch). Without
			// eviction the dict grows by every marker that ever existed in
			// the session and the per-render-frame mutation loop keeps
			// shifting stale sprites that are no longer drawn.
			public long LastRefreshedTicks;
		}

		internal enum MessageKind
		{
			None = 0,
			Atlas = 1,
			DrawString = 2,
			DrawStringAligned = 3,
			// MyRenderMessageDrawSprite (the pixel-rect path used by the
			// `string` overload of MyHudMarkerRenderBase.AddTexturedQuad —
			// DrawIcon emits the inner GPS-marker pin texture
			// "marker_gps.dds" via this path).
			Sprite = 4
		}

		public static void SetContext(
			Vector3D worldPos,
			Vector2 vanillaScreenPos,
			bool isOffScreenArrow,
			Vector2 oldEllipsePosition,
			bool isCrosshair = false)
		{
			_currentContext = new MarkerContext
			{
				WorldPos = worldPos,
				VanillaScreenPos = vanillaScreenPos,
				IsOffScreenArrow = isOffScreenArrow,
				OldEllipsePosition = oldEllipsePosition,
				IsCrosshair = isCrosshair
			};
		}

		public static void ClearContext()
		{
			_currentContext = null;
		}

		public static void OnHudTextStart(MyHudText text)
		{
			if (_currentContext == null)
			{
				_hudTextContexts.Remove(text);
				return;
			}

			_hudTextContexts.Remove(text);
			_hudTextContexts.Add(text, _currentContext);
		}

		public static bool TryGetHudTextContext(MyHudText text, out MarkerContext ctx)
		{
			return _hudTextContexts.TryGetValue(text, out ctx);
		}

		// Render-thread postfix on MyMessagePool.Return. Drops the
		// disposed message's entry from _trackedSprites BEFORE the pool
		// can hand the instance back to sim's MessagePool.Get on a later
		// tick. Closing this window prevents the 1-frame swap of a
		// marker slot with another HUD element's content (e.g. placement-
		// mode bind-shortcut prompt): without it, sim's next-tick
		// MessagePool.Get on this pooled instance hands back a message
		// still tagged in our dictionary from a past marker emission,
		// and render's per-frame refresh can iterate _trackedSprites
		// between sim writing the new content's Position and our
		// EnqueueMessage postfix dropping the stale entry — overwriting
		// the new content's Position with the marker's smoothed Position
		// so the visible slot draws the new content at the marker's
		// screen position.
		public static void OnMessagePoolReturn(MyRenderMessageBase message)
		{
			if (_trackedSprites.IsEmpty)
			{
				return;
			}
			MarkerContext _;
			_trackedSprites.TryRemove(message, out _);
		}

		public static void OnMessageEnqueued(MyRenderMessageBase message)
		{
			if (_currentContext == null)
			{
				// Non-marker emission. PatchMyMessagePoolReturn already
				// cleared any stale marker entry when this pooled
				// instance was disposed back to the pool, so nothing to
				// do here.
				return;
			}

			// Build a fresh per-message context — the OriginalPosition captured
			// here is sticky for this message's lifetime in the dictionary.
			// Creating a new instance avoids sharing VanillaScreenPos with
			// siblings (each marker's atlas vs text have the same
			// VanillaScreenPos but different originals).
			MessageKind kind;
			Vector2 origAtlas = default;
			Vector2 origString = default;
			Vector2 origSpriteRect = default;

			// Order matters: MyRenderMessageDrawStringAligned derives from
			// MyRenderMessageDrawString, so the more-specific case has to
			// come first or the base case would shadow it.
			switch (message)
			{
				case MyRenderMessageDrawSpriteAtlas atlas:
					kind = MessageKind.Atlas;
					origAtlas = atlas.Position;
					break;
				case MyRenderMessageDrawStringAligned strAligned:
					kind = MessageKind.DrawStringAligned;
					origString = strAligned.ScreenCoord;
					break;
				case MyRenderMessageDrawString str:
					kind = MessageKind.DrawString;
					origString = str.ScreenCoord;
					break;
				case MyRenderMessageDrawSprite sprite:
					kind = MessageKind.Sprite;
					origSpriteRect = new Vector2(
						sprite.DestinationRectangle.X,
						sprite.DestinationRectangle.Y);
					break;
				default:
					return;
			}

			var ctx = new MarkerContext
			{
				WorldPos = _currentContext.WorldPos,
				VanillaScreenPos = _currentContext.VanillaScreenPos,
				OriginalAtlasPosition = origAtlas,
				OriginalStringScreenCoord = origString,
				OriginalSpriteRectPos = origSpriteRect,
				Kind = kind,
				IsOffScreenArrow = _currentContext.IsOffScreenArrow,
				OldEllipsePosition = _currentContext.OldEllipsePosition,
				IsCrosshair = _currentContext.IsCrosshair,
				LastRefreshedTicks = Stopwatch.GetTimestamp()
			};

			// Pool reuse: same MyRenderMessageBase instance may be re-used
			// across sim ticks. On re-emission, OnMessageEnqueued fires again
			// and we overwrite the entry with fresh OriginalPosition values.
			// The render-thread refresh loop is iterator-safe with
			// ConcurrentDictionary updates.
			_trackedSprites[message] = ctx;
		}

		// Called per render frame from RenderFrameSmoothing.RunFrame. Iterates
		// every tracked message and overwrites its position field with
		// (original + smoothed delta) computed against this render frame's
		// smoothed camera.
		public static void RefreshAllTrackedSpritesPerRenderFrame()
		{
			if (_trackedSprites.IsEmpty)
			{
				return;
			}

			var now = Stopwatch.GetTimestamp();
			_evictionScratch.Clear();

			foreach (var kv in _trackedSprites)
			{
				if (now - kv.Value.LastRefreshedTicks > _staleThresholdTicks)
				{
					_evictionScratch.Add(kv.Key);
					continue;
				}

				ApplyMutation(kv.Key, kv.Value);
			}

			for (var i = 0; i < _evictionScratch.Count; i++)
			{
				MarkerContext _;
				_trackedSprites.TryRemove(_evictionScratch[i], out _);
			}
		}

		private static void ApplyMutation(MyRenderMessageBase message, MarkerContext ctx)
		{
			if (ctx.IsCrosshair)
			{
				ApplyCrosshairMutation(message, ctx);
				return;
			}

			var worldPos = ResolveSmoothedWorldPos(ctx);

			if (!TryReprojectSmoothed(worldPos, out var smoothedScreen))
			{
				return;
			}

			if (ctx.IsOffScreenArrow)
			{
				ApplyOffScreenArrowMutation(message, ctx, smoothedScreen);
				return;
			}

			var deltaNormalized = smoothedScreen - ctx.VanillaScreenPos;

			switch (ctx.Kind)
			{
				case MessageKind.Atlas:
				{
					if (!(message is MyRenderMessageDrawSpriteAtlas atlas))
					{
						return;
					}
					// SET (not increment) so the mutation across render frames
					// doesn't accumulate. Atlas Position is in HUD-coord space
					// (= normalized × MyGuiManager.GetHudSize() because
					// PointOfInterest.Draw multiplies projectedPoint2D by
					// hudSize before passing to AddTexturedQuad). Engine
					// downstream multiplies Position × Scale = Position ×
					// (safeFullscreen/hudSize) to land in pixel space, so a
					// pixel shift of `deltaNormalized × safeFullscreen`
					// requires a Position shift of `deltaNormalized ×
					// hudSize`.
					var hudSize = MyGuiManager.GetHudSize();
					atlas.Position = ctx.OriginalAtlasPosition + deltaNormalized * hudSize;
					return;
				}
				case MessageKind.DrawString:
				{
					if (!(message is MyRenderMessageDrawString str))
					{
						return;
					}

					var deltaPixel = NormalizedDeltaToPixels(deltaNormalized);
					str.ScreenCoord = ctx.OriginalStringScreenCoord + deltaPixel;
					return;
				}
				case MessageKind.DrawStringAligned:
				{
					if (!(message is MyRenderMessageDrawStringAligned strAligned))
					{
						return;
					}

					var deltaPixel = NormalizedDeltaToPixels(deltaNormalized);
					strAligned.ScreenCoord = ctx.OriginalStringScreenCoord + deltaPixel;
					return;
				}
				case MessageKind.Sprite:
				{
					if (!(message is MyRenderMessageDrawSprite sprite))
					{
						return;
					}
					// DestinationRectangle is a pixel-space RectangleF.
					// AddTexturedQuad's string overload converts the
					// projectedPoint2D from HUD-coord to pixels via
					// `position × safeFullscreen/hudSize` which equals
					// `normalized × safeFullscreen` — same scale as the other
					// two text paths after the engine's full pipeline runs.
					var deltaPixel = NormalizedDeltaToPixels(deltaNormalized);
					sprite.DestinationRectangle.X = ctx.OriginalSpriteRectPos.X + deltaPixel.X;
					sprite.DestinationRectangle.Y = ctx.OriginalSpriteRectPos.Y + deltaPixel.Y;
					return;
				}
			}
		}

		// Crosshair mutation. The crosshair anchor is `seat.GetPosition() +
		// 1000 * seat.Forward` — a "ship-rigid" point, but ~1000 m outside
		// the ship's bounding sphere, so the moving-grid sphere registry
		// can't find it. Use the dedicated piloted-ship slot
		// (`BillboardCorrections.TryGetPilotedShipCorrection`) instead. The
		// world coord is FRESH at sim T (computed inline in
		// MyShipController.DrawHud, not pulled from a stale POI field), so
		// the current-pose correction `inv(grid_T) * grid_smoothed` is the
		// right inverse to use here — different from POI markers, which need
		// the prev-pose variant.
		//
		// The crosshair atlas Position is in MIXED units: X is normalized
		// projected coord [0..1] (post triple-head adjustment), Y is the
		// normalized Y multiplied by `MyGuiManager.GetHudSize().Y`. So the
		// per-frame delta has different scales per axis: X scales by 1, Y
		// scales by HudSize.Y.
		private static void ApplyCrosshairMutation(MyRenderMessageBase message, MarkerContext ctx)
		{
			if (ctx.Kind != MessageKind.Atlas)
			{
				return;
			}
			if (!(message is MyRenderMessageDrawSpriteAtlas atlas))
			{
				return;
			}

			if (!BillboardCorrections.TryGetPilotedShipCorrection(out var correction))
			{
				return;
			}

			var worldPosCorrected = Vector3D.Transform(ctx.WorldPos, correction);

			if (!TryReprojectSmoothed(worldPosCorrected, out var smoothedScreen))
			{
				return;
			}

			// Match MyHudCrosshair's storage convention: X normalized
			// (m_position.X), Y multiplied by hudSize.Y (m_position.Y). The
			// atlas's stored OriginalAtlasPosition already has the
			// triple-head `+1` adjustment baked in (from MyHudCrosshair.Draw)
			// — preserve it by adding the delta rather than overwriting
			// Position outright.
			var hudSize = MyGuiManager.GetHudSize();
			var deltaCrosshair = new Vector2(
				smoothedScreen.X - ctx.VanillaScreenPos.X,
				(smoothedScreen.Y - ctx.VanillaScreenPos.Y) * hudSize.Y);

			atlas.Position = ctx.OriginalAtlasPosition + deltaCrosshair;
		}

		// Off-screen-arrow mutation. Vanilla `PointOfInterest.Draw` clamps the
		// arrow to a 0.77-radius ellipse around hud center pointing toward
		// the marker. The on-screen `Position += deltaNormalized × hudSize`
		// shape doesn't reproduce that — the engine's clamp is non-linear in
		// the projection (renormalize → place on ellipse), so a delta
		// computed in raw-projection space lands the arrow in the wrong spot
		// once you re-clamp. Instead we recompute the ellipse-clamped
		// position directly from the smoothed-camera projection, then build
		// the per-render-frame shift as `new_ellipse - OldEllipsePosition`
		// and apply it to each tagged sprite (atlas position, text screen
		// coords). For the atlas we also rebuild `RightVector` from the new
		// radial direction so the arrow stays oriented outward.
		private static void ApplyOffScreenArrowMutation(
			MyRenderMessageBase message, MarkerContext ctx, Vector2 smoothedScreen)
		{
			var hudSize = MyGuiManager.GetHudSize();
			var hudSizeHalf = MyGuiManager.GetHudSizeHalf();
			var uiScale = MyGuiManager.UIScale;

			var smoothedHud = smoothedScreen * hudSize;
			var smoothedVec = smoothedHud - hudSizeHalf;
			var smoothedDir = smoothedVec.LengthSquared() < 1e-10f
				? new Vector2(1f, 0f)
				: Vector2.Normalize(smoothedVec);
			var newEllipse = hudSizeHalf + hudSizeHalf * smoothedDir * 0.77f * uiScale;

			var shiftHud = newEllipse - ctx.OldEllipsePosition;

			switch (ctx.Kind)
			{
				case MessageKind.Atlas:
				{
					if (!(message is MyRenderMessageDrawSpriteAtlas atlas))
					{
						return;
					}

					atlas.Position = ctx.OriginalAtlasPosition + shiftHud;
					// AddTexturedQuad derives `rightVector = (-up.Y, up.X)` from
					// the upVector it's passed (90° CCW rotation in
					// MyHudMarkerRenderBase.AddTexturedQuad). For the
					// off-screen arrow that upVector is the radial direction
					// from hud center to the marker. Rebuild from the smoothed
					// direction so the arrow points the right way as it
					// slides along the screen edge.
					atlas.RightVector = new Vector2(-smoothedDir.Y, smoothedDir.X);
					return;
				}
				case MessageKind.DrawString:
				{
					if (!(message is MyRenderMessageDrawString str))
					{
						return;
					}

					var shiftPixel = NormalizedDeltaToPixels(shiftHud / hudSize);
					str.ScreenCoord = ctx.OriginalStringScreenCoord + shiftPixel;
					return;
				}
				case MessageKind.DrawStringAligned:
				{
					if (!(message is MyRenderMessageDrawStringAligned strAligned))
					{
						return;
					}

					var shiftPixel = NormalizedDeltaToPixels(shiftHud / hudSize);
					strAligned.ScreenCoord = ctx.OriginalStringScreenCoord + shiftPixel;
					return;
				}
				case MessageKind.Sprite:
				{
					// The inner GPS-pin DrawSprite path is only emitted by
					// PointOfInterest.DrawIcon (the on-screen icon path), so
					// off-screen arrows shouldn't tag a Sprite. Skip
					// defensively rather than apply a wrong shift.
					return;
				}
			}
		}

		// Convert a normalized [0..1] delta into a pixel-space delta that
		// matches where the engine's atlas / DrawSprite / DrawString
		// pipelines actually land sprites on screen
		// (`pixel = normalized × safeFullscreen` after the full chain in
		// each path — verified by running the round-trip math through
		// `MyGuiScreenHudBase.ConvertHudToNormalizedGuiPosition` →
		// `MyGuiManager.GetScreenCoordinateFromNormalizedCoordinate` for the
		// text path; the atlas and DrawSprite paths reach the same pixel
		// position via `position × safeFullscreen/hudSize`).
		// `useFullClientArea: true` makes
		// `GetScreenCoordinateFromNormalizedCoordinate` use
		// `m_safeFullscreenRectangle` instead of `m_safeGuiRectangle`, which
		// is the right rectangle for this calculation.
		private static Vector2 NormalizedDeltaToPixels(Vector2 deltaNormalized)
		{
			var origin = MyGuiManager.GetScreenCoordinateFromNormalizedCoordinate(Vector2.Zero, true);
			var shifted = MyGuiManager.GetScreenCoordinateFromNormalizedCoordinate(deltaNormalized, true);
			return shifted - origin;
		}

		// Route the marker's world coord through any smoothed grid whose
		// vanilla bounding sphere contains it, applying that grid's
		// vanilla→smoothed rigid correction. Fixes the piloting case where
		// the marker's W is rebaked at sim rate from a moving entity
		// (broadcast antenna, block info on a strafing ship): the correction
		// moves W onto the grid's smoothed pose so projecting it through the
		// smoothed camera lands at the same view-space position the grid
		// mesh is rendered at. Static GPS markers (W far from any registered
		// grid) miss the lookup and fall through to ctx.WorldPos unchanged.
		//
		// Uses the *prev-pose* correction (`inv(grid_T-1) * grid_smoothed`),
		// not the current-pose one. Empirically the marker's POI
		// WorldPosition is one sim tick stale — a ship-attached marker's W
		// reflects the antenna's world coord at *grid_T-1*, while
		// CameraHistory's `curr` snapshot holds grid_T. Applying
		// `inv(grid_T)` to a grid_T-1-frame point produces an off-by-one
		// drift exactly equal to the ship's per-tick motion in the opposite
		// direction (visible as "strafe left, marker drifts right by
		// per_tick" — confirmed against the antenna block). Using the
		// inverse of the *previous* pose maps the stale W onto the smoothed
		// pose correctly, with the marker overlapping the rendered block.
		//
		// Same registry the billboard rebake's line/point paths use;
		// RenderFrameSmoothing.RunFrame already populates it per render
		// frame, before this method is reached via
		// RefreshAllTrackedSpritesPerRenderFrame.
		private static Vector3D ResolveSmoothedWorldPos(MarkerContext ctx)
		{
			return BillboardCorrections.TryFindPrevPoseCorrection(ctx.WorldPos, out var correction)
				? Vector3D.Transform(ctx.WorldPos, correction)
				: ctx.WorldPos;
		}

		// Project worldPos through the smoothed camera at the current render
		// timestamp, returning HUD coords [0..1]. Math mirrors vanilla
		// MyHudMarkerRender.TryComputeScreenPoint line-for-line: full view
		// matrix, Vector4D.Transform (no auto perspective divide),
		// `position.Z > 0` sign-flip mirroring for behind-camera points,
		// explicit W check + manual perspective divide.
		// Out-of-reasonable-range result returns false (no mutation that
		// frame) to keep the icon stable through projection singularities.
		private static bool TryReprojectSmoothed(Vector3D worldPos, out Vector2 hudCoord)
		{
			hudCoord = Vector2.Zero;

			if (!RenderFrameSmoothing.TryGetSmoothedPose(out var camPos, out var camRot))
			{
				return false;
			}

			if (!CameraHistory.TryGetPair(out _, out var curr))
			{
				return false;
			}

			var rotMatF = Matrix.CreateFromQuaternion(camRot);
			var worldMat = (MatrixD)rotMatF;
			worldMat.Translation = camPos;
			MatrixD.Invert(ref worldMat, out var viewMat);

			var posInView = Vector3D.Transform(worldPos, viewMat);
			var clip = Vector4D.Transform(posInView, (MatrixD)curr.ProjectionMatrix);

			if (posInView.Z > 0.0)
			{
				clip.X = -clip.X;
				clip.Y = -clip.Y;
			}

			if (clip.W == 0.0)
			{
				return false;
			}

			var x = clip.X / clip.W / 2.0 + 0.5;
			var y = -clip.Y / clip.W / 2.0 + 0.5;

			if (x < -2.0 || x > 3.0 || y < -2.0 || y > 3.0)
			{
				return false;
			}

			hudCoord = new Vector2((float)x, (float)y);
			return true;
		}
	}

	/// <summary>
	///     Sim-thread postfix on <see cref="MyHudMarkerRender.TryComputeScreenPoint"/>.
	/// </summary>
	[HarmonyPatch(typeof(MyHudMarkerRender), "TryComputeScreenPoint")]
	public static class PatchTryComputeScreenPoint
	{
		public static void Postfix(Vector3D worldPosition, Vector2 projectedPoint2D, bool isBehind, bool __result)
		{
			// Only project markers drawn from within MyHudMarkerRender.Draw.
			// Out-of-Draw callers (grid target reticule, etc.) would otherwise
			// leak this context onto the next HUD emission. Clear any stale
			// context so a leaked one can't survive into the toolbar draw.
			if (!HudMarkerSmoothing.InMarkerDraw)
			{
				HudMarkerSmoothing.ClearContext();
				return;
			}

			if (!__result || isBehind)
			{
				HudMarkerSmoothing.ClearContext();
				return;
			}

			// Replicates the off-screen-arrow selector inside
			// MyHudMarkerRender.PointOfInterest.Draw. projectedPoint2D from
			// TryComputeScreenPoint is in normalized HUD coords [0..1]; the
			// engine multiplies by hudSize before the branch test, so we do
			// the same. The vanilla condition also includes `viewSpace.Z >
			// 0.0`, but we already filter that out via the `isBehind`
			// early-return above (close enough — the dot-product `isBehind`
			// flags markers more than 90° off-axis, which covers the
			// view-space-Z-positive case for any marker the user can see).
			var hudSize = MyGuiManager.GetHudSize();
			var hudSizeHalf = MyGuiManager.GetHudSizeHalf();
			var uiScale = MyGuiManager.UIScale;
			var inset = hudSize * (1f - uiScale) / 2f + new Vector2(0.04f, 0.04f);
			var hudPos = projectedPoint2D * hudSize;
			var isOffScreenArrow =
				hudPos.X < inset.X || hudPos.X > hudSize.X - inset.X ||
				hudPos.Y < inset.Y || hudPos.Y > hudSize.Y - inset.Y;

			// Engine's clamp inside PointOfInterest.Draw: place the icon on a
			// 0.77-radius ellipse around hud center, in the direction of the
			// marker. Replicate so we have a reference for the
			// per-render-frame shift (the actual atlas Position the engine
			// sets is also this value, modulo any AddTexturedQuad
			// adjustments — close enough for the shift basis).
			var oldEllipse = Vector2.Zero;
			if (isOffScreenArrow)
			{
				var vec = hudPos - hudSizeHalf;
				var dir = vec.LengthSquared() < 1e-10f
					? new Vector2(1f, 0f)
					: Vector2.Normalize(vec);
				oldEllipse = hudSizeHalf + hudSizeHalf * dir * 0.77f * uiScale;
			}

			HudMarkerSmoothing.SetContext(worldPosition, projectedPoint2D, isOffScreenArrow, oldEllipse);
		}
	}

	/// <summary>
	///     Sim-thread postfix on <see cref="MyHudMarkerRender"/><c>.Draw</c>.
	/// </summary>
	[HarmonyPatch(typeof(MyHudMarkerRender), "Draw")]
	public static class PatchMyHudMarkerRenderDraw
	{
		public static void Prefix()
		{
			HudMarkerSmoothing.InMarkerDraw = true;
		}

		public static void Postfix()
		{
			HudMarkerSmoothing.InMarkerDraw = false;
			HudMarkerSmoothing.ClearContext();
		}
	}

	/// <summary>
	///     Sim-thread postfix on private <c>MyRenderProxy.EnqueueMessage</c>.
	///     Tags any sprite message enqueued while a marker context is set.
	/// </summary>
	[HarmonyPatch]
	public static class PatchEnqueueMessage
	{
		public static MethodBase TargetMethod()
		{
			return AccessTools.Method(typeof(MyRenderProxy), "EnqueueMessage")
				?? throw Errors.NotResolved("MyRenderProxy.EnqueueMessage");
		}

		public static void Postfix(MyRenderMessageBase message)
		{
			HudMarkerSmoothing.OnMessageEnqueued(message);
			ArtificialHorizonSmoothing.OnMessageEnqueued(message);
		}
	}

	/// <summary>
	///     Render-thread postfix on
	///     <see cref="MyMessagePool"/><c>.Return</c>, the engine's
	///     pooled-message dispose path. Drops the disposed message's
	///     entries from every consumer that keys state by
	///     <see cref="MyRenderMessageBase"/> reference, before sim's
	///     next-tick <c>MyMessagePool.Get</c> can hand the instance back
	///     to a different consumer with a stale tag still attached.
	///
	///     Picked over the generic <c>Get&lt;T&gt;</c> side because
	///     <c>Return</c> takes <see cref="MyRenderMessageBase"/> by
	///     non-generic parameter — one stable patch site instead of
	///     four closed-generic instantiations whose JIT lifecycle is
	///     fragile under Harmony.
	/// </summary>
	[HarmonyPatch]
	public static class PatchMyMessagePoolReturn
	{
		public static MethodBase TargetMethod()
		{
			return AccessTools.Method(typeof(MyMessagePool), "Return",
					new[] { typeof(MyRenderMessageBase) })
				?? throw Errors.NotResolved("VRageRender.MyMessagePool.Return");
		}

		public static void Postfix(MyRenderMessageBase message)
		{
			HudMarkerSmoothing.OnMessagePoolReturn(message);
			ArtificialHorizonSmoothing.OnMessagePoolReturn(message);
		}
	}

	/// <summary>
	///     Sim-thread postfix on <see cref="MyHudText.Start"/>. Binds the
	///     active marker context (set by
	///     <see cref="PatchTryComputeScreenPoint"/>) to the MyHudText
	///     instance via a side
	///     <see cref="ConditionalWeakTable{TKey, TValue}"/>. The actual
	///     <c>DrawString</c> emission for these texts happens later from
	///     <see cref="MyGuiScreenHudBase"/><c>.DrawTexts</c>, which runs
	///     after our marker-draw postfix has cleared the thread-static
	///     context — so we stash the context here on the per-text object and
	///     look it up at draw time in <see cref="PatchDrawTexts"/>.
	/// </summary>
	[HarmonyPatch(typeof(MyHudText), "Start")]
	public static class PatchMyHudTextStart
	{
		public static void Postfix(MyHudText __instance)
		{
			HudMarkerSmoothing.OnHudTextStart(__instance);
		}
	}

	/// <summary>
	///     Replaces <see cref="MyGuiScreenHudBase"/><c>.DrawTexts</c>'s
	///     iteration with one that re-installs each text's saved marker
	///     context immediately before the inner
	///     <see cref="MyGuiManager"/><c>.DrawString</c> call, so the
	///     resulting <see cref="MyRenderMessageDrawString"/>
	///     gets tagged by <see cref="PatchEnqueueMessage"/> with the marker's
	///     world coord. The original loop runs on the sim thread with the
	///     marker context already cleared (the marker-render iteration
	///     finished some time before <c>DrawTexts</c>), so without this
	///     replacement no text-label messages would be tagged at all.
	///
	///     Body matches the engine's <c>DrawTexts</c> verbatim except for
	///     the bracketing <c>SetContext</c>/<c>ClearContext</c> calls.
	///     Helper methods (<c>ConvertHudToNormalizedGuiPosition</c>,
	///     <c>MyGuiManager.MeasureString</c>,
	///     <c>MyUtils.GetCoordTopLeftFromAligned</c>,
	///     <c>MyGuiTextShadows.DrawShadow</c>,
	///     <c>MyGuiManager.DrawString</c>) are public; only the
	///     <c>m_texts</c> field needs reflection.
	/// </summary>
	[HarmonyPatch(typeof(MyGuiScreenHudBase), "DrawTexts")]
	public static class PatchDrawTexts
	{
		private static readonly FieldInfo _textsField = AccessTools.Field(typeof(MyGuiScreenHudBase), "m_texts")
			?? throw Errors.NotResolved("MyGuiScreenHudBase.m_texts");

		public static bool Prefix(MyGuiScreenHudBase __instance)
		{
			if (!(_textsField.GetValue(__instance) is MyObjectsPoolSimple<MyHudText> pool))
			{
				return true;
			}

			var count = pool.GetAllocatedCount();
			if (count <= 0)
			{
				return false;
			}

			for (var i = 0; i < count; i++)
			{
				var item = pool.GetAllocatedItem(i);
				var sb = item.GetStringBuilder();
				if (sb.Length == 0)
				{
					continue;
				}

				var hasMarker = HudMarkerSmoothing.TryGetHudTextContext(item, out var markerCtx);
				if (hasMarker)
				{
					HudMarkerSmoothing.SetContext(
						markerCtx.WorldPos,
						markerCtx.VanillaScreenPos,
						markerCtx.IsOffScreenArrow,
						markerCtx.OldEllipsePosition);
				}
				else
				{
					HudMarkerSmoothing.ClearContext();
				}

				try
				{
					var font = item.Font;
					item.Position /= MyGuiManager.GetHudSize();
					var alignedCoord = MyGuiScreenHudBase.ConvertHudToNormalizedGuiPosition(ref item.Position);
					var textSize = MyGuiManager.MeasureString(font, sb, MyGuiSandbox.GetDefaultTextScaleWithLanguage());
					textSize *= item.Scale;
					alignedCoord = MyUtils.GetCoordTopLeftFromAligned(alignedCoord, textSize, item.Alignement);
					MyGuiTextShadows.DrawShadow(ref alignedCoord, ref textSize, null,
						item.Color.A / 255f,
						MyGuiDrawAlignEnum.HORISONTAL_LEFT_AND_VERTICAL_TOP, true);
					MyGuiManager.DrawString(font, sb.ToString(), alignedCoord, item.Scale, item.Color,
						MyGuiDrawAlignEnum.HORISONTAL_LEFT_AND_VERTICAL_TOP, false,
						float.PositiveInfinity, true);
				}
				finally
				{
					HudMarkerSmoothing.ClearContext();
				}
			}

			pool.ClearAllAllocated();
			return false;
		}
	}

	/// <summary>
	///     Hooks <see cref="MyHudCrosshair"/><c>.Draw</c> to tag the
	///     piloting-crosshair atlas message with a marker context whose
	///     world anchor is `seat.GetPosition() + 1000 * seat.Forward` — the
	///     same world point <c>MyShipController.DrawHud</c> uses to compute
	///     <c>m_position</c> via <see cref="MyHudCrosshair"/><c>.GetProjectedVector</c>.
	///     <see cref="HudMarkerSmoothing"/>'s mutation path then routes the
	///     tag through <c>BillboardCorrections.TryGetPilotedShipCorrection</c>
	///     (current-pose, sphere-free lookup) and re-projects the corrected
	///     world coord through the smoothed camera per render frame, so the
	///     crosshair tracks the ship's smoothed forward direction instead of
	///     stuttering at sim rate.
	///
	///     Gated to piloting only: when the controlled entity isn't a
	///     <see cref="MyShipController"/> (on-foot character, spectator),
	///     the prefix returns without setting context, so the on-screen
	///     crosshair sprite still emits but isn't smoothed (no behavior
	///     change for those modes — the crosshair is centered and stable
	///     anyway).
	/// </summary>
	[HarmonyPatch(typeof(MyHudCrosshair), "Draw")]
	public static class PatchMyHudCrosshairDraw
	{
		public static void Prefix(MyHudCrosshair __instance)
		{
			var session = MySession.Static;
			var controlledEntity = session?.ControlledEntity?.Entity;
			if (controlledEntity == null)
			{
				return;
			}

			if (!(controlledEntity is MyShipController shipController))
			{
				return;
			}

			var positionComp = shipController.PositionComp;
			if (positionComp == null)
			{
				return;
			}

			// Mirrors MyShipController.DrawHud:
			//   worldPosition = base.PositionComp.GetPosition()
			//                 + 1000.0 * base.PositionComp.WorldMatrixRef.Forward
			var worldPos = positionComp.GetPosition()
				+ 1000.0 * positionComp.WorldMatrixRef.Forward;

			// MyHudCrosshair.m_position storage convention:
			//   X = projected.X / W / 2 + 0.5  (normalized [0..1], post
			//                                   triple-head adjust)
			//   Y = (-projected.Y / W / 2 + 0.5) * hudSize.Y  (HUD-pixel)
			// Convert to fully-normalized for the standard reproject delta
			// math; the crosshair-specific mutation path in HudMarkerSmoothing
			// re-applies the Y * hudSize.Y scale before writing back to
			// atlas.Position.
			var hudSize = MyGuiManager.GetHudSize();
			var hudYScale = hudSize.Y > 0f ? hudSize.Y : 1f;
			var crosshairPos = __instance.Position;
			var vanillaNormalized = new Vector2(crosshairPos.X, crosshairPos.Y / hudYScale);

			HudMarkerSmoothing.SetContext(
				worldPos,
				vanillaNormalized,
				isOffScreenArrow: false,
				oldEllipsePosition: Vector2.Zero,
				isCrosshair: true);
		}

		public static void Postfix()
		{
			HudMarkerSmoothing.ClearContext();
		}
	}
}
