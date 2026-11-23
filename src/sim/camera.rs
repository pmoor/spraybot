use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    anti_alias::smaa::Smaa,
    app::AppExit,
    camera::{Hdr, RenderTarget, ShadowLodOrigin},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    light::VolumetricFog,
    pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel},
    post_process::{
        bloom::Bloom,
        dof::{DepthOfField, DepthOfFieldMode},
    },
    prelude::*,
    render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection, Msaa},
};

use crate::cli::CameraPreset;
use crate::core::types::JobOp;
use crate::sim::recorder::GracefulShutdownRequested;
use crate::sim::state::{AppConfig, OffscreenRenderTarget, SimClock, SimStepSet, SprayJobRes};

/// Studio interior camera clamp bounds (meters, 1 unit = 1m).
const STUDIO_MIN_X: f32 = -3.85;
const STUDIO_MAX_X: f32 = 3.85;
const STUDIO_MIN_Y: f32 = 0.08;
const STUDIO_MAX_Y: f32 = 4.05;
const STUDIO_MIN_Z: f32 = 0.12;
const STUDIO_MAX_Z: f32 = 6.30;

const MIN_PITCH: f32 = -FRAC_PI_2 + 0.05;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.05;
const MIN_DISTANCE: f32 = 0.45;
const MAX_DISTANCE: f32 = 7.50;

/// Marker component for the main 3D studio camera.
#[derive(Component)]
pub struct MainStudioCamera;

/// Marker component for the root node of the live-view Studio HUD.
#[derive(Component)]
struct StudioHudRoot;

/// Marker component for the live-view Studio HUD text block.
#[derive(Component)]
struct StudioHudText;

/// Controls whether the live-view Studio HUD overlay is visible (`H` hotkey).
#[derive(Resource, Debug, Clone, Copy)]
struct HudVisible(bool);

impl Default for HudVisible {
    fn default() -> Self {
        Self(true)
    }
}

/// Orbit/pan/zoom/follow state for the studio camera.
#[derive(Resource, Debug, Clone)]
pub struct CameraRig {
    pub active_preset: Option<CameraPreset>,
    pub desired_target: Vec3,
    pub current_target: Vec3,
    pub desired_yaw: f32,
    pub current_yaw: f32,
    pub desired_pitch: f32,
    pub current_pitch: f32,
    pub desired_distance: f32,
    pub current_distance: f32,
}

impl CameraRig {
    /// Builds an initial rig matching a CLI [`CameraPreset`] and initial TCP position.
    pub fn from_preset(preset: CameraPreset, initial_nozzle: Vec3) -> Self {
        let (eye, target) = preset_eye_target(preset, initial_nozzle);
        let (yaw, pitch, dist) = cartesian_to_spherical(eye, target);
        Self {
            active_preset: Some(preset),
            desired_target: target,
            current_target: target,
            desired_yaw: yaw,
            current_yaw: yaw,
            desired_pitch: pitch,
            current_pitch: pitch,
            desired_distance: dist,
            current_distance: dist,
        }
    }

    /// Transitions smoothly to one of the 4 named camera presets.
    pub fn apply_preset(&mut self, preset: CameraPreset, current_nozzle: Vec3) {
        let (eye, target) = preset_eye_target(preset, current_nozzle);
        let (yaw, pitch, dist) = cartesian_to_spherical(eye, target);
        self.active_preset = Some(preset);
        self.desired_target = target;
        self.desired_yaw = unwrap_angle_near(yaw, self.current_yaw);
        self.desired_pitch = pitch.clamp(MIN_PITCH, MAX_PITCH);
        self.desired_distance = dist.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    /// Updates Follow target tracking (holding `desired_target` frozen while `is_shaking`
    /// is true) and advances exponential smoothing for target, orbit, and zoom over `dt`.
    pub fn step_tracking(&mut self, nozzle_world_pos: Vec3, is_shaking: bool, dt: f32) {
        // Freeze desired_target during ShakeCan so the Follow camera stays rock steady.
        if self.active_preset == Some(CameraPreset::Follow) && !is_shaking {
            self.desired_target = clamp_target_inside_studio(nozzle_world_pos);
        }

        // Softer target damping in Follow mode for smooth sweeping paint strokes;
        // faster damping for discrete preset jumps and manual pan.
        let target_rate = if self.active_preset == Some(CameraPreset::Follow) {
            4.5
        } else {
            8.0
        };
        let target_alpha = 1.0 - (-target_rate * dt).exp();
        let orbit_alpha = 1.0 - (-16.0 * dt).exp();

        self.current_target = self.current_target.lerp(self.desired_target, target_alpha);
        self.current_yaw += (self.desired_yaw - self.current_yaw) * orbit_alpha;
        self.current_pitch += (self.desired_pitch - self.current_pitch) * orbit_alpha;
        self.current_distance += (self.desired_distance - self.current_distance) * orbit_alpha;
    }

    /// Computes the camera world eye position and clamps it inside the studio walls (`Y >= 0.08m`).
    pub fn clamped_eye(&self) -> Vec3 {
        let raw_eye = spherical_to_cartesian(
            self.current_target,
            self.current_yaw,
            self.current_pitch,
            self.current_distance,
        );
        clamp_inside_studio(raw_eye)
    }
}

/// Registers camera spawning, hotkey input handling, mouse orbit/pan/zoom, 3D gizmos, and live HUD.
pub fn plugin(app: &mut App) {
    app.init_resource::<HudVisible>()
        .add_systems(Startup, (spawn_camera, spawn_studio_hud).chain())
        .add_systems(Update, handle_hotkeys.in_set(SimStepSet::InputAndClock))
        .add_systems(
            Update,
            (update_camera_rig, draw_debug_gizmos, update_studio_hud)
                .chain()
                .in_set(SimStepSet::CameraAndGizmos),
        );
}

/// Returns photographic DepthOfField settings appropriate for the active camera preset,
/// keeping `Canvas` completely crisp edge-to-edge (`None`).
fn dof_for_preset(preset: Option<CameraPreset>, distance: f32) -> Option<DepthOfField> {
    match preset {
        Some(CameraPreset::Canvas) => None,
        Some(CameraPreset::Follow) | Some(CameraPreset::Rack) => Some(DepthOfField {
            mode: DepthOfFieldMode::Gaussian,
            focal_distance: distance.max(0.45),
            sensor_height: 0.01866,
            aperture_f_stops: 5.6,
            max_circle_of_confusion_diameter: 9.0,
            max_depth: 12.0,
        }),
        Some(CameraPreset::Overview) | None => Some(DepthOfField {
            mode: DepthOfFieldMode::Gaussian,
            focal_distance: distance.max(0.6),
            sensor_height: 0.01866,
            aperture_f_stops: 11.0,
            max_circle_of_confusion_diameter: 4.0,
            max_depth: 14.0,
        }),
    }
}

fn spawn_camera(
    mut commands: Commands,
    app_config: Res<AppConfig>,
    offscreen: Res<OffscreenRenderTarget>,
    job: Res<SprayJobRes>,
) {
    let rig = CameraRig::from_preset(app_config.cli.camera, job.current_frame.nozzle_world_pos);
    let eye = rig.clamped_eye();
    let transform = Transform::from_translation(eye).looking_at(rig.current_target, Vec3::Y);

    let mut bloom = Bloom::NATURAL;
    bloom.intensity = 0.065;
    bloom.low_frequency_boost = 0.45;
    bloom.high_pass_frequency = 0.90;

    let color_grading = ColorGrading {
        global: ColorGradingGlobal {
            exposure: 0.04,
            post_saturation: 1.03,
            ..default()
        },
        shadows: ColorGradingSection {
            contrast: 1.01,
            ..default()
        },
        highlights: ColorGradingSection {
            gamma: 0.98,
            ..default()
        },
        ..default()
    };

    let mut entity = commands.spawn((
        MainStudioCamera,
        Camera3d::default(),
        ShadowLodOrigin,
        Hdr,
        Msaa::Off,
        DepthPrepass,
        DeferredPrepass,
        Smaa::default(),
        ScreenSpaceAmbientOcclusion {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::High,
            radius: 0.30,
            constant_object_thickness: 0.08,
        },
        // Deferred prepass keeps Bevy 0.20.1 clearcoat normals intact alongside `SSAO`
        // (`NormalPrepass`), while omitting `ScreenSpaceReflections` prevents stochastic static
        // reflection flicker (`GeneratedEnvironmentMapLight` + PBR lights drive reflections).
        VolumetricFog {
            ambient_color: Color::srgb(0.86, 0.90, 0.96),
            ambient_intensity: 0.02,
            step_count: 48,
            jitter: 0.0,
        },
        Tonemapping::AcesFitted,
        color_grading,
        bloom,
        transform,
    ));

    if let Some(dof) = dof_for_preset(rig.active_preset, rig.current_distance) {
        entity.insert(dof);
    }

    if let Some(image_handle) = offscreen.image.as_ref() {
        entity.insert(RenderTarget::Image(image_handle.clone().into()));
    }

    commands.insert_resource(rig);
}

fn spawn_studio_hud(
    mut commands: Commands,
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
    job: Res<SprayJobRes>,
    rig: Res<CameraRig>,
) {
    if app_config.cli.record.is_some() {
        return;
    }

    let initial_text = format_studio_hud_text(&clock, &job, &rig);
    commands.spawn((
        StudioHudRoot,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
            border_radius: BorderRadius::all(Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.05, 0.06, 0.08, 0.78)),
        Visibility::Inherited,
        children![(
            StudioHudText,
            Text::new(initial_text),
            TextFont::from_font_size(13.5),
            TextColor(Srgba::hex("#E8ECF2").unwrap().into()),
        )],
    ));
}

#[allow(clippy::too_many_arguments)]
fn handle_hotkeys(
    keys: Res<ButtonInput<KeyCode>>,
    app_config: Res<AppConfig>,
    mut clock: ResMut<SimClock>,
    mut job: ResMut<SprayJobRes>,
    mut rig: ResMut<CameraRig>,
    mut hud_visible: ResMut<HudVisible>,
    mut shutdown_req: ResMut<GracefulShutdownRequested>,
    mut app_exit: MessageWriter<AppExit>,
) {
    let nozzle = job.current_frame.nozzle_world_pos;

    if keys.just_pressed(KeyCode::Digit1) || keys.just_pressed(KeyCode::Numpad1) {
        rig.apply_preset(CameraPreset::Overview, nozzle);
    } else if keys.just_pressed(KeyCode::Digit2) || keys.just_pressed(KeyCode::Numpad2) {
        rig.apply_preset(CameraPreset::Follow, nozzle);
    } else if keys.just_pressed(KeyCode::Digit3) || keys.just_pressed(KeyCode::Numpad3) {
        rig.apply_preset(CameraPreset::Rack, nozzle);
    } else if keys.just_pressed(KeyCode::Digit4) || keys.just_pressed(KeyCode::Numpad4) {
        rig.apply_preset(CameraPreset::Canvas, nozzle);
    }

    if keys.just_pressed(KeyCode::Space) {
        clock.paused = !clock.paused;
    }

    if keys.just_pressed(KeyCode::BracketLeft) {
        clock.speed = (clock.speed * 0.5).clamp(0.25, 32.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        clock.speed = (clock.speed * 2.0).clamp(0.25, 32.0);
    }

    if keys.just_pressed(KeyCode::KeyG) {
        clock.show_gizmos = !clock.show_gizmos;
        job.show_gizmos = clock.show_gizmos;
    }

    if app_config.cli.record.is_none() && keys.just_pressed(KeyCode::KeyH) {
        hud_visible.0 = !hud_visible.0;
    }

    if keys.just_pressed(KeyCode::KeyR) {
        clock.reset();
        job.current_frame = job.timeline.sample(0.0);
    }

    if keys.just_pressed(KeyCode::Escape) || keys.just_pressed(KeyCode::KeyQ) {
        if app_config.cli.record.is_some() {
            shutdown_req.0 = true;
        } else {
            app_exit.write(AppExit::Success);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_camera_rig(
    mut commands: Commands,
    time: Res<Time>,
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
    job: Res<SprayJobRes>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_reader: MessageReader<MouseMotion>,
    mut wheel_reader: MessageReader<MouseWheel>,
    mut rig: ResMut<CameraRig>,
    mut cam_query: Query<(Entity, &mut Transform, Option<&mut DepthOfField>), With<MainStudioCamera>>,
) {
    let mut mouse_delta = Vec2::ZERO;
    for ev in motion_reader.read() {
        mouse_delta += ev.delta;
    }

    let mut scroll_y = 0.0_f32;
    for ev in wheel_reader.read() {
        scroll_y += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y * 0.02,
        };
    }

    // Left-drag: orbit azimuth and elevation.
    if mouse_buttons.pressed(MouseButton::Left) && mouse_delta != Vec2::ZERO {
        rig.desired_yaw -= mouse_delta.x * 0.0055;
        rig.desired_pitch = (rig.desired_pitch + mouse_delta.y * 0.0055).clamp(MIN_PITCH, MAX_PITCH);
    }

    // Right/Middle-drag: pan focal target along camera plane.
    if (mouse_buttons.pressed(MouseButton::Right) || mouse_buttons.pressed(MouseButton::Middle))
        && mouse_delta != Vec2::ZERO
    {
        if rig.active_preset == Some(CameraPreset::Follow) {
            rig.active_preset = None;
        }
        let eye = rig.clamped_eye();
        let forward = (rig.current_target - eye).normalize_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward).normalize_or(Vec3::Y);
        let pan_scale = rig.current_distance * 0.0014;
        rig.desired_target += (-right * mouse_delta.x + up * mouse_delta.y) * pan_scale;
        rig.desired_target = clamp_target_inside_studio(rig.desired_target);
    }

    // Scroll wheel: smooth exponential dolly zoom.
    if scroll_y.abs() > 1e-5 {
        rig.desired_distance =
            (rig.desired_distance * (-scroll_y * 0.12).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    // Use deterministic step dt during recording so follow-cam motion is lockstep-smooth.
    let dt = if app_config.cli.record.is_some() {
        let fps = app_config.cli.fps.max(1) as f32;
        if clock.paused {
            0.0
        } else {
            (1.0 / fps) * clock.speed.clamp(0.25, 4.0)
        }
    } else {
        time.delta_secs().min(0.1)
    };

    rig.step_tracking(
        job.current_frame.nozzle_world_pos,
        job.current_frame.is_shaking,
        dt,
    );

    let clamped_eye = rig.clamped_eye();
    let desired_dof = dof_for_preset(rig.active_preset, rig.current_distance);
    for (cam_entity, mut transform, dof_opt) in cam_query.iter_mut() {
        *transform = Transform::from_translation(clamped_eye).looking_at(rig.current_target, Vec3::Y);
        match (desired_dof, dof_opt) {
            (Some(target_dof), Some(mut cur_dof)) => {
                cur_dof.focal_distance = target_dof.focal_distance;
                cur_dof.aperture_f_stops = target_dof.aperture_f_stops;
                cur_dof.max_circle_of_confusion_diameter =
                    target_dof.max_circle_of_confusion_diameter;
                cur_dof.max_depth = target_dof.max_depth;
            }
            (Some(target_dof), None) => {
                commands.entity(cam_entity).insert(target_dof);
            }
            (None, Some(_)) => {
                commands.entity(cam_entity).remove::<DepthOfField>();
            }
            (None, None) => {}
        }
    }
}

fn draw_debug_gizmos(mut gizmos: Gizmos, clock: Res<SimClock>, job: Res<SprayJobRes>) {
    if !clock.show_gizmos && !job.show_gizmos {
        return;
    }

    // 1. Planned spray strokes on the wall canvas plane (slightly offset in +Z to prevent z-fighting).
    let active_stroke_color = Color::srgba(0.15, 0.90, 1.0, 0.9);
    let lead_color = Color::srgba(1.0, 0.65, 0.15, 0.45);
    for op in &job.job.ops {
        if let JobOp::PaintStroke { points_canvas_3d } = op {
            for pair in points_canvas_3d.windows(2) {
                let a = pair[0];
                let b = pair[1];
                let pa = a.pos + a.normal * 0.004;
                let pb = b.pos + b.normal * 0.004;
                let color = if a.valve_on && b.valve_on {
                    active_stroke_color
                } else {
                    lead_color
                };
                gizmos.line(pa, pb, color);
            }
        }
    }

    // 2. Active nozzle / TCP coordinate axes.
    let frame = &job.current_frame;
    let apex = frame.nozzle_world_pos;
    let dir = frame.nozzle_world_dir.normalize_or(Vec3::NEG_Z);
    let tcp_rot = Quat::from_rotation_arc(Vec3::NEG_Z, dir);
    gizmos.axes(Transform::from_translation(apex).with_rotation(tcp_rot), 0.10);

    // 3. Dynamic cap-aware spray cone wireframe from nozzle tip toward canvas (Z = 0.0).
    let cone_len = if dir.z < -1e-3 && apex.z > 0.0 {
        (apex.z / -dir.z).clamp(0.04, 0.35)
    } else {
        0.22
    };
    let cone_radius = cone_len * frame.active_cap.cone_half_angle_rad().tan();
    let cone_base_center = apex + dir * cone_len;
    let cone_color = if frame.spray_active {
        Color::srgba(0.25, 1.0, 0.40, 0.95)
    } else {
        Color::srgba(0.55, 0.70, 0.90, 0.55)
    };

    let ring_rot = Quat::from_rotation_arc(Vec3::Z, dir);
    gizmos
        .circle(Isometry3d::new(cone_base_center, ring_rot), cone_radius, cone_color)
        .resolution(24);

    let right = ring_rot * Vec3::X;
    let up = ring_rot * Vec3::Y;
    for i in 0..8 {
        let angle = (i as f32) * (2.0 * PI / 8.0);
        let (sin_a, cos_a) = angle.sin_cos();
        let rim_pt = cone_base_center + (right * cos_a + up * sin_a) * cone_radius;
        gizmos.line(apex, rim_pt, cone_color);
    }
}

fn update_studio_hud(
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
    job: Res<SprayJobRes>,
    rig: Res<CameraRig>,
    hud_visible: Res<HudVisible>,
    mut root_q: Query<&mut Visibility, With<StudioHudRoot>>,
    mut text_q: Query<&mut Text, With<StudioHudText>>,
) {
    if app_config.cli.record.is_some() {
        return;
    }

    let vis = if hud_visible.0 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut root_vis in &mut root_q {
        *root_vis = vis;
    }

    if hud_visible.0 {
        let formatted = format_studio_hud_text(&clock, &job, &rig);
        for mut text in &mut text_q {
            text.0 = formatted.clone();
        }
    }
}

#[inline]
fn format_mm_ss(secs: f64) -> String {
    let total = secs.max(0.0).floor() as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Formats the 3-line Live-View Studio HUD string from the current playback, camera, and cap state.
pub fn format_studio_hud_text(clock: &SimClock, job: &SprayJobRes, rig: &CameraRig) -> String {
    let status = if clock.finished {
        "DONE"
    } else if clock.paused {
        "PAUSED"
    } else {
        "PLAYING"
    };
    let pct = if clock.total_duration > 1e-6 {
        ((clock.t_sim / clock.total_duration) * 100.0)
            .clamp(0.0, 100.0)
            .round() as u32
    } else {
        100
    };
    let line1 = format!(
        "{status}  |  {:.2}x  |  {} / {} ({pct}%)",
        clock.speed,
        format_mm_ss(clock.t_sim),
        format_mm_ss(clock.total_duration),
    );

    let preset_str = match rig.active_preset {
        Some(CameraPreset::Overview) => "Overview",
        Some(CameraPreset::Follow) => "Follow",
        Some(CameraPreset::Rack) => "Rack",
        Some(CameraPreset::Canvas) => "Canvas",
        None => "Free",
    };
    let frame = &job.current_frame;
    let can_str = match frame.held_can_slot {
        Some(slot) => match job.palette.cans.get(slot) {
            Some(can) => format!(
                "Can #{slot} (#{:02X}{:02X}{:02X})",
                can.color_srgb[0], can.color_srgb[1], can.color_srgb[2]
            ),
            None => format!("Can #{slot}"),
        },
        None => "None".to_string(),
    };
    let gizmos_str = if clock.show_gizmos {
        "Gizmos: ON"
    } else {
        "Gizmos: OFF"
    };
    let line2 = format!(
        "{preset_str}  |  {can_str}  |  {}  |  {gizmos_str}",
        frame.active_cap.label()
    );
    let line3 =
        "[ / ] Speed  |  Space Pause  |  1-4 Cam  |  G Gizmos  |  H HUD  |  R Reset  |  Q Quit";

    format!("{line1}\n{line2}\n{line3}")
}

/// Returns `(eye, target)` world coordinates for each named [`CameraPreset`].
pub fn preset_eye_target(preset: CameraPreset, nozzle: Vec3) -> (Vec3, Vec3) {
    match preset {
        CameraPreset::Overview => (Vec3::new(1.65, 1.85, 3.35), Vec3::new(-0.35, 1.35, 0.25)),
        CameraPreset::Follow => {
            let target = clamp_target_inside_studio(nozzle);
            let eye = clamp_inside_studio(target + Vec3::new(0.45, 0.32, 1.35));
            (eye, target)
        }
        CameraPreset::Rack => (Vec3::new(-1.05, 1.35, 1.85), Vec3::new(-1.85, 1.08, 0.60)),
        CameraPreset::Canvas => (Vec3::new(0.0, 1.55, 2.35), Vec3::new(0.0, 1.55, 0.0)),
    }
}

/// Converts `(eye, target)` into spherical `(yaw, pitch, distance)` around `target`.
pub fn cartesian_to_spherical(eye: Vec3, target: Vec3) -> (f32, f32, f32) {
    let offset = eye - target;
    let distance = offset.length().clamp(MIN_DISTANCE, MAX_DISTANCE);
    let horiz = (offset.x * offset.x + offset.z * offset.z).sqrt();
    let yaw = offset.x.atan2(offset.z);
    let pitch = offset.y.atan2(horiz).clamp(MIN_PITCH, MAX_PITCH);
    (yaw, pitch, distance)
}

/// Converts spherical `(yaw, pitch, distance)` around `target` back into an eye position.
pub fn spherical_to_cartesian(target: Vec3, yaw: f32, pitch: f32, distance: f32) -> Vec3 {
    let (sin_p, cos_p) = pitch.sin_cos();
    let (sin_y, cos_y) = yaw.sin_cos();
    let dir = Vec3::new(cos_p * sin_y, sin_p, cos_p * cos_y);
    target + dir * distance
}

/// Unwraps `target_angle` by multiples of `2π` to stay closest to `reference` (avoids 360° spins).
fn unwrap_angle_near(target_angle: f32, reference: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let delta = (target_angle - reference).rem_euclid(two_pi);
    let shortest = if delta > PI { delta - two_pi } else { delta };
    reference + shortest
}

/// Clamps camera eye position inside the studio room walls and above the floor (`Y >= 0.08m`).
pub fn clamp_inside_studio(eye: Vec3) -> Vec3 {
    Vec3::new(
        eye.x.clamp(STUDIO_MIN_X, STUDIO_MAX_X),
        eye.y.clamp(STUDIO_MIN_Y, STUDIO_MAX_Y),
        eye.z.clamp(STUDIO_MIN_Z, STUDIO_MAX_Z),
    )
}

fn clamp_target_inside_studio(target: Vec3) -> Vec3 {
    Vec3::new(
        target.x.clamp(-3.50, 3.50),
        target.y.clamp(0.05, 3.80),
        target.z.clamp(-0.02, 5.80),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spherical_cartesian_roundtrip_for_all_presets() {
        let nozzle = Vec3::new(0.25, 1.45, 0.14);
        for preset in [
            CameraPreset::Overview,
            CameraPreset::Follow,
            CameraPreset::Rack,
            CameraPreset::Canvas,
        ] {
            let (eye, target) = preset_eye_target(preset, nozzle);
            let (yaw, pitch, dist) = cartesian_to_spherical(eye, target);
            let reconstructed = spherical_to_cartesian(target, yaw, pitch, dist);
            assert_relative_eq!(reconstructed.x, eye.x, epsilon = 1e-4);
            assert_relative_eq!(reconstructed.y, eye.y, epsilon = 1e-4);
            assert_relative_eq!(reconstructed.z, eye.z, epsilon = 1e-4);

            let rig = CameraRig::from_preset(preset, nozzle);
            let clamped = rig.clamped_eye();
            assert!(clamped.y >= STUDIO_MIN_Y);
            assert!(clamped.x >= STUDIO_MIN_X && clamped.x <= STUDIO_MAX_X);
            assert!(clamped.z >= STUDIO_MIN_Z && clamped.z <= STUDIO_MAX_Z);
        }
    }

    #[test]
    fn floor_and_wall_clamping_enforces_studio_bounds() {
        let below_floor = clamp_inside_studio(Vec3::new(0.0, -1.5, 2.0));
        assert_relative_eq!(below_floor.y, 0.08, epsilon = 1e-6);

        let outside_walls = clamp_inside_studio(Vec3::new(10.0, 9.0, -2.0));
        assert_relative_eq!(outside_walls.x, STUDIO_MAX_X, epsilon = 1e-6);
        assert_relative_eq!(outside_walls.y, STUDIO_MAX_Y, epsilon = 1e-6);
        assert_relative_eq!(outside_walls.z, STUDIO_MIN_Z, epsilon = 1e-6);
    }

    #[test]
    fn studio_hud_text_formats_playback_can_cap_and_controls() {
        use crate::core::types::{NozzleCap, SprayCanSpec, StudioConfig};

        let studio = StudioConfig::default();
        let mut job = SprayJobRes::empty(&studio);
        job.palette.cans.push(SprayCanSpec {
            slot_index: 2,
            color_srgb: [0x1A, 0x7F, 0xE8],
            label: "Cobalt".to_string(),
        });
        job.palette.cans.resize(
            3,
            SprayCanSpec {
                slot_index: 2,
                color_srgb: [0x1A, 0x7F, 0xE8],
                label: "Cobalt".to_string(),
            },
        );
        job.current_frame.held_can_slot = Some(2);
        job.current_frame.active_cap = NozzleCap::Skinny;

        let mut clock = SimClock {
            t_sim: 65.0,
            total_duration: 130.0,
            speed: 2.0,
            paused: true,
            show_gizmos: true,
            ..Default::default()
        };
        let mut rig = CameraRig::from_preset(CameraPreset::Follow, Vec3::ZERO);
        let text = format_studio_hud_text(&clock, &job, &rig);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("PAUSED"));
        assert!(lines[0].contains("2.00x"));
        assert!(lines[0].contains("01:05 / 02:10 (50%)"));
        assert!(lines[1].contains("Follow"));
        assert!(lines[1].contains("Can #2 (#1A7FE8)"));
        assert!(lines[1].contains("Skinny Cap (2.6°)"));
        assert!(lines[1].contains("Gizmos: ON"));
        assert!(lines[2].contains("H HUD"));

        clock.paused = false;
        clock.finished = true;
        clock.show_gizmos = false;
        rig.active_preset = None;
        job.current_frame.held_can_slot = None;
        job.current_frame.active_cap = NozzleCap::Fat;
        let text2 = format_studio_hud_text(&clock, &job, &rig);
        assert!(text2.contains("DONE"));
        assert!(text2.contains("Free"));
        assert!(text2.contains("None"));
        assert!(text2.contains("Fat Cap (12°)"));
        assert!(text2.contains("Gizmos: OFF"));
    }

    #[test]
    fn follow_camera_freezes_target_during_shake_and_tracks_smoothly_when_not_shaking() {
        let initial_nozzle = Vec3::new(-0.40, 1.35, 0.45);
        let mut rig = CameraRig::from_preset(CameraPreset::Follow, initial_nozzle);
        let pre_shake_target = rig.current_target;
        assert_relative_eq!(pre_shake_target.x, initial_nozzle.x, epsilon = 1e-6);
        assert_relative_eq!(pre_shake_target.y, initial_nozzle.y, epsilon = 1e-6);
        assert_relative_eq!(pre_shake_target.z, initial_nozzle.z, epsilon = 1e-6);

        // During ShakeCan (is_shaking == true), oscillating nozzle must NOT move desired_target or current_target.
        // Orbit/zoom should still respond smoothly while target stays frozen.
        rig.desired_yaw += 0.25;
        for step in 0..30 {
            let phase = (step as f32) * 0.45;
            let shaking_nozzle = initial_nozzle + Vec3::new(0.06 * phase.sin(), 0.12 * phase.cos(), 0.04 * phase.sin());
            rig.step_tracking(shaking_nozzle, true, 1.0 / 60.0);
            assert_relative_eq!(rig.desired_target.x, pre_shake_target.x, epsilon = 1e-6);
            assert_relative_eq!(rig.desired_target.y, pre_shake_target.y, epsilon = 1e-6);
            assert_relative_eq!(rig.desired_target.z, pre_shake_target.z, epsilon = 1e-6);
            assert_relative_eq!(rig.current_target.x, pre_shake_target.x, epsilon = 1e-6);
            assert_relative_eq!(rig.current_target.y, pre_shake_target.y, epsilon = 1e-6);
            assert_relative_eq!(rig.current_target.z, pre_shake_target.z, epsilon = 1e-6);
        }
        assert!(
            (rig.current_yaw - rig.desired_yaw).abs() < 0.01,
            "orbit should remain responsive while shake freezes target"
        );

        // After ShakeCan ends (is_shaking == false), Follow smoothly tracks the new nozzle position
        // using the softer 4.5 exponential rate (not snapping instantaneously).
        let target_nozzle = Vec3::new(0.45, 1.65, 0.16);
        let dt = 1.0 / 60.0;
        rig.step_tracking(target_nozzle, false, dt);
        assert_relative_eq!(rig.desired_target.x, target_nozzle.x, epsilon = 1e-6);
        assert_relative_eq!(rig.desired_target.y, target_nozzle.y, epsilon = 1e-6);
        assert_relative_eq!(rig.desired_target.z, target_nozzle.z, epsilon = 1e-6);

        let expected_alpha = 1.0 - (-4.5 * dt).exp();
        let expected_after_one = pre_shake_target.lerp(target_nozzle, expected_alpha);
        assert_relative_eq!(rig.current_target.x, expected_after_one.x, epsilon = 1e-5);
        assert_relative_eq!(rig.current_target.y, expected_after_one.y, epsilon = 1e-5);
        assert_relative_eq!(rig.current_target.z, expected_after_one.z, epsilon = 1e-5);

        // After ~1.5s of steady tracking, camera converges smoothly near target_nozzle
        for _ in 0..90 {
            rig.step_tracking(target_nozzle, false, dt);
        }
        assert!((rig.current_target - target_nozzle).length() < 0.005);
    }

    #[test]
    fn discrete_preset_jump_uses_faster_damping_than_follow_tracking() {
        let start = Vec3::new(0.0, 1.5, 0.2);
        let dest = Vec3::new(1.0, 1.5, 0.2);
        let dt = 1.0 / 60.0;

        let mut follow_rig = CameraRig::from_preset(CameraPreset::Follow, start);
        follow_rig.step_tracking(dest, false, dt);
        let follow_progress = follow_rig.current_target.x;

        let mut overview_rig = CameraRig::from_preset(CameraPreset::Follow, start);
        overview_rig.apply_preset(CameraPreset::Overview, start);
        overview_rig.desired_target = dest;
        overview_rig.step_tracking(dest, false, dt);
        let preset_progress = overview_rig.current_target.x;

        assert!(
            preset_progress > follow_progress * 1.4,
            "discrete preset jump ({preset_progress}) should damp faster than Follow ({follow_progress})"
        );
    }
}
