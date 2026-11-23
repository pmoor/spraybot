use std::f32::consts::PI;

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::core::types::NozzleCap;
#[cfg(test)]
use crate::sim::canvas::CONE_HALF_ANGLE_RAD;
use crate::sim::state::{AppConfig, SimClock, SimStepSet, SprayJobRes};

/// Maximum pre-allocated mist billboard quads in the pool.
const MIST_POOL_SIZE: usize = 256;

/// Target puff spawn rate per second of simulation time while spraying.
const MIST_SPAWN_RATE_HZ: f32 = 160.0;

/// Hard cap on newly activated puffs per rendered frame (keeps `--speed 15+` crisp without quad floods).
const MAX_SPAWNS_PER_FRAME: usize = 16;

/// Quadratic aerodynamic drag coefficient for mist puffs in air.
const AIR_DRAG_COEFF: f32 = 6.5;

/// Maximum pre-allocated ambient studio dust mote billboards in the pool.
const DUST_MOTE_POOL_SIZE: usize = 48;

/// Registers the volumetric aerosol mist cone billboard FX systems.
pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_mist)
        .add_systems(Update, update_mist.in_set(SimStepSet::CanvasAndFx));
}

#[derive(Resource)]
struct MistState {
    last_reset_epoch: u64,
    spawn_accum: f32,
    rng: DeterministicRng,
    cursor: usize,
    entities: Vec<Entity>,
}

#[derive(Component)]
struct MistParticle {
    active: bool,
    age: f32,
    lifetime: f32,
    pos: Vec3,
    vel: Vec3,
    start_size: f32,
    end_size: f32,
    color_srgb: [u8; 3],
    base_alpha: f32,
    roll: f32,
    roll_vel: f32,
    mat_handle: Handle<StandardMaterial>,
}

#[derive(Component)]
struct DustMote {
    base_pos: Vec3,
    amp: Vec3,
    freq: Vec3,
    phase: f32,
    size: f32,
    base_alpha: f32,
    mat_handle: Handle<StandardMaterial>,
}

/// Lightweight deterministic SplitMix64 PRNG for mist puff jitter.
#[derive(Debug, Clone)]
pub struct DeterministicRng(u64);

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1_u32 << 24) as f32)
    }

    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

fn setup_mist(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    app_config: Res<AppConfig>,
    clock: Res<SimClock>,
) {
    let mist_tex = images.add(build_soft_radial_mist_texture(128));
    let quad_mesh = meshes.add(Rectangle::new(1.0, 1.0));

    let mut entities = Vec::with_capacity(MIST_POOL_SIZE);
    for _ in 0..MIST_POOL_SIZE {
        let mat_handle = materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 1.0, 1.0, 0.0),
            base_color_texture: Some(mist_tex.clone()),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        });

        let entity = commands
            .spawn((
                Name::new("MistPuff"),
                Mesh3d(quad_mesh.clone()),
                MeshMaterial3d(mat_handle.clone()),
                Transform::from_scale(Vec3::splat(0.02)),
                Visibility::Hidden,
                NotShadowCaster,
                NotShadowReceiver,
                MistParticle {
                    active: false,
                    age: 0.0,
                    lifetime: 0.25,
                    pos: Vec3::ZERO,
                    vel: Vec3::ZERO,
                    start_size: 0.014,
                    end_size: 0.075,
                    color_srgb: [255, 255, 255],
                    base_alpha: 0.22,
                    roll: 0.0,
                    roll_vel: 0.0,
                    mat_handle,
                },
            ))
            .id();
        entities.push(entity);
    }

    commands.insert_resource(MistState {
        last_reset_epoch: clock.reset_epoch,
        spawn_accum: 0.0,
        rng: DeterministicRng::new(app_config.cli.seed.wrapping_add(1337)),
        cursor: 0,
        entities,
    });

    // Deterministic pool of subtle floating dust motes catching studio light cones
    let mut dust_rng = DeterministicRng::new(app_config.cli.seed.wrapping_add(7331));
    for _ in 0..DUST_MOTE_POOL_SIZE {
        let base_alpha = dust_rng.range(0.04, 0.11);
        let mat_handle = materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.96, 0.90, base_alpha),
            base_color_texture: Some(mist_tex.clone()),
            alpha_mode: AlphaMode::Add,
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        });
        let base_pos = Vec3::new(
            dust_rng.range(-2.6, 2.2),
            dust_rng.range(1.1, 3.3),
            dust_rng.range(0.45, 2.8),
        );
        let size = dust_rng.range(0.005, 0.009);
        commands.spawn((
            Name::new("StudioDustMote"),
            Mesh3d(quad_mesh.clone()),
            MeshMaterial3d(mat_handle.clone()),
            Transform::from_translation(base_pos).with_scale(Vec3::splat(size)),
            Visibility::Inherited,
            NotShadowCaster,
            NotShadowReceiver,
            DustMote {
                base_pos,
                amp: Vec3::new(
                    dust_rng.range(0.03, 0.09),
                    dust_rng.range(0.02, 0.07),
                    dust_rng.range(0.03, 0.08),
                ),
                freq: Vec3::new(
                    dust_rng.range(0.08, 0.22),
                    dust_rng.range(0.06, 0.18),
                    dust_rng.range(0.07, 0.20),
                ),
                phase: dust_rng.range(0.0, 2.0 * PI),
                size,
                base_alpha,
                mat_handle,
            },
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn update_mist(
    clock: Res<SimClock>,
    spray_job: Res<SprayJobRes>,
    app_config: Res<AppConfig>,
    mut mist_state: ResMut<MistState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera_q: Query<&GlobalTransform, With<Camera3d>>,
    mut particle_q: Query<(&mut MistParticle, &mut Transform, &mut Visibility), Without<DustMote>>,
    mut dust_q: Query<(&DustMote, &mut Transform), Without<MistParticle>>,
) {
    let t_sim_f32 = clock.t_sim as f32;
    let cam_rot = camera_q
        .iter()
        .next()
        .map(|gtf| gtf.to_scale_rotation_translation().1)
        .unwrap_or(Quat::IDENTITY);

    for (mote, mut tf) in &mut dust_q {
        let p = mote.phase;
        let offset = Vec3::new(
            (t_sim_f32 * mote.freq.x + p).sin() * mote.amp.x,
            (t_sim_f32 * mote.freq.y + p * 1.37).cos() * mote.amp.y,
            (t_sim_f32 * mote.freq.z + p * 0.71).sin() * mote.amp.z,
        );
        tf.translation = mote.base_pos + offset;
        tf.rotation = cam_rot;
        tf.scale = Vec3::splat(mote.size);
        let glint = 0.65 + 0.35 * (t_sim_f32 * 0.55 + p * 2.1).sin();
        if let Some(mut mat) = materials.get_mut(&mote.mat_handle) {
            mat.base_color = Color::srgba(1.0, 0.96, 0.90, mote.base_alpha * glint);
        }
    }

    if clock.reset_epoch != mist_state.last_reset_epoch {
        mist_state.last_reset_epoch = clock.reset_epoch;
        mist_state.spawn_accum = 0.0;
        for (mut p, _, mut vis) in &mut particle_q {
            p.active = false;
            *vis = Visibility::Hidden;
        }
        return;
    }

    let dt_sim = (clock.t_sim - clock.prev_t_sim).max(0.0) as f32;
    let wall_z = app_config.studio.canvas_center.z + 0.008;

    // 1. Advance active mist particles
    if dt_sim > 0.0 {
        for (mut p, mut tf, mut vis) in &mut particle_q {
            if !p.active {
                continue;
            }
            p.age += dt_sim;
            if p.age >= p.lifetime {
                p.active = false;
                *vis = Visibility::Hidden;
                continue;
            }

            step_particle_physics(&mut p, dt_sim, wall_z);
            apply_particle_visual(&p, cam_rot, &mut tf, &mut vis, &mut materials);
        }
    }

    // 2. Spawn new mist puffs when spray is active and simulation time advanced
    let frame = &spray_job.current_frame;
    if dt_sim <= 0.0 || !frame.spray_active {
        if !frame.spray_active {
            mist_state.spawn_accum = 0.0;
        }
        return;
    }

    let desired = mist_state.spawn_accum + MIST_SPAWN_RATE_HZ * dt_sim;
    let spawn_count = (desired.floor() as usize).min(MAX_SPAWNS_PER_FRAME);
    mist_state.spawn_accum = if desired >= MAX_SPAWNS_PER_FRAME as f32 {
        0.0
    } else {
        desired - (spawn_count as f32)
    };

    if spawn_count == 0 {
        return;
    }

    let can_srgb = frame
        .held_can_slot
        .and_then(|slot| spray_job.palette.cans.get(slot))
        .map(|can| can.color_srgb)
        .unwrap_or([245, 245, 245]);

    let nozzle_dir = frame.nozzle_world_dir.normalize_or(Vec3::NEG_Z);
    let nozzle_pos = frame.nozzle_world_pos;

    // On large timelapse steps, pre-age puffs slightly along the cone so the plume looks continuous.
    let pre_age_max = (dt_sim * 0.5).min(0.08);
    let pool_len = mist_state.entities.len();
    let cone_half_angle = frame.active_cap.cone_half_angle_rad();
    let (size_scale, alpha_scale) = if frame.active_cap == NozzleCap::Skinny {
        (0.48_f32, 0.80_f32)
    } else {
        (1.0_f32, 1.0_f32)
    };

    for i in 0..spawn_count {
        let mut spawned = false;
        for attempt in 0..pool_len {
            let idx = (mist_state.cursor + attempt) % pool_len;
            let entity = mist_state.entities[idx];
            if let Ok((mut p, mut tf, mut vis)) = particle_q.get_mut(entity)
                && (!p.active || attempt + 1 == pool_len)
            {
                let emit_dir = sample_cone_direction(nozzle_dir, cone_half_angle, &mut mist_state.rng);
                let speed = mist_state.rng.range(1.65, 2.55);
                let lifetime = mist_state.rng.range(0.18, 0.35);
                let initial_age = if dt_sim > 0.04 {
                    (i as f32 / spawn_count as f32) * pre_age_max
                } else {
                    0.0
                };

                p.active = true;
                p.age = initial_age.min(lifetime * 0.6);
                p.lifetime = lifetime;
                p.pos = nozzle_pos + emit_dir * mist_state.rng.range(0.004, 0.016);
                p.vel = emit_dir * speed;
                p.start_size = mist_state.rng.range(0.014, 0.022) * size_scale;
                p.end_size = mist_state.rng.range(0.060, 0.092) * size_scale;
                p.color_srgb = can_srgb;
                p.base_alpha =
                    mist_state.rng.range(0.16, 0.28) * frame.flow_scale.clamp(0.35, 1.0) * alpha_scale;
                p.roll = mist_state.rng.range(0.0, 2.0 * PI);
                p.roll_vel = mist_state.rng.range(-2.5, 2.5);

                if initial_age > 0.0 {
                    step_particle_physics(&mut p, initial_age, wall_z);
                }
                apply_particle_visual(&p, cam_rot, &mut tf, &mut vis, &mut materials);

                mist_state.cursor = (idx + 1) % pool_len;
                spawned = true;
                break;
            }
        }
        if !spawned {
            break;
        }
    }
}

fn step_particle_physics(p: &mut MistParticle, dt: f32, wall_z: f32) {
    let sub_steps = ((dt / 0.02).ceil() as usize).clamp(1, 8);
    let h = dt / (sub_steps as f32);

    for _ in 0..sub_steps {
        let speed = p.vel.length();
        if speed > 1e-4 {
            let drag_dv = (AIR_DRAG_COEFF * speed * speed * h).min(speed * 0.90);
            p.vel -= p.vel * (drag_dv / speed);
        }
        // Subtle upward thermal drift as mist slows
        p.vel.y += 0.12 * h;
        p.pos += p.vel * h;
        p.roll += p.roll_vel * h;

        if p.pos.z < wall_z {
            p.pos.z = wall_z;
            if p.vel.z < 0.0 {
                let impact = -p.vel.z;
                p.vel.z = 0.0;
                // Gentle radial splash along the wall surface
                let xy = Vec2::new(p.vel.x, p.vel.y);
                let dir_xy = if xy.length_squared() > 1e-5 {
                    xy.normalize()
                } else {
                    Vec2::new(p.roll.cos(), p.roll.sin())
                };
                p.vel.x += dir_xy.x * impact * 0.30;
                p.vel.y += dir_xy.y * impact * 0.30;
            }
        }
    }
}

fn apply_particle_visual(
    p: &MistParticle,
    cam_rot: Quat,
    tf: &mut Transform,
    vis: &mut Visibility,
    materials: &mut Assets<StandardMaterial>,
) {
    let u = (p.age / p.lifetime).clamp(0.0, 1.0);
    let size = p.start_size + (p.end_size - p.start_size) * (1.0 - (1.0 - u).powi(2));
    let fade_in = (u / 0.12).clamp(0.0, 1.0);
    let fade_out = (1.0 - u).powf(1.65);
    let alpha = (p.base_alpha * fade_in * fade_out).clamp(0.0, 1.0);

    tf.translation = p.pos;
    tf.rotation = cam_rot * Quat::from_rotation_z(p.roll);
    tf.scale = Vec3::splat(size);
    *vis = if alpha > 0.004 {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };

    if let Some(mut mat) = materials.get_mut(&p.mat_handle) {
        mat.base_color = Color::srgba(
            p.color_srgb[0] as f32 / 255.0,
            p.color_srgb[1] as f32 / 255.0,
            p.color_srgb[2] as f32 / 255.0,
            alpha,
        );
    }
}

/// Samples a unit ray direction uniformly distributed across a cone of half-angle `half_angle_rad`.
pub fn sample_cone_direction(
    axis: Vec3,
    half_angle_rad: f32,
    rng: &mut DeterministicRng,
) -> Vec3 {
    let dir = axis.normalize_or(Vec3::NEG_Z);
    let helper = if dir.y.abs() < 0.95 { Vec3::Y } else { Vec3::X };
    let u_axis = dir.cross(helper).normalize();
    let v_axis = dir.cross(u_axis).normalize();

    let phi = rng.range(0.0, 2.0 * PI);
    let r = rng.next_f32().sqrt();
    let theta = half_angle_rad * r;
    (dir * theta.cos() + (u_axis * phi.cos() + v_axis * phi.sin()) * theta.sin()).normalize()
}

/// Generates a soft radial-alpha wispy mist puff texture (`Rgba8UnormSrgb`).
fn build_soft_radial_mist_texture(size: u32) -> Image {
    let size = size.max(16);
    let mut data = vec![255_u8; (size as usize) * (size as usize) * 4];
    let center = (size as f32 - 1.0) * 0.5;
    let inv_radius = 1.0 / center.max(1.0);

    for py in 0..size {
        for px in 0..size {
            let dx = (px as f32 - center) * inv_radius;
            let dy = (py as f32 - center) * inv_radius;
            let r = (dx * dx + dy * dy).sqrt();

            let alpha = if r >= 1.0 {
                0.0
            } else {
                let angle = dy.atan2(dx);
                let wisp = 0.88
                    + 0.08 * (angle * 3.0 + r * 4.0).sin()
                    + 0.04 * (angle * 5.0 - r * 6.0).cos();
                let radial = ((1.0 - r * r).max(0.0)).powf(2.2);
                (radial * wisp).clamp(0.0, 1.0)
            };

            let idx = ((py as usize) * (size as usize) + (px as usize)) * 4;
            data[idx] = 255;
            data[idx + 1] = 255;
            data[idx + 2] = 255;
            data[idx + 3] = (alpha * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mist_cone_samples_stay_within_12_degrees() {
        let mut rng = DeterministicRng::new(42);
        let axis = Vec3::NEG_Z;
        let max_cos = CONE_HALF_ANGLE_RAD.cos() - 1e-5;
        for _ in 0..500 {
            let d = sample_cone_direction(axis, CONE_HALF_ANGLE_RAD, &mut rng);
            assert!((d.length() - 1.0).abs() < 1e-5);
            assert!(
                d.dot(axis) >= max_cos,
                "sampled mist ray exceeded 12 deg cone: dot={}",
                d.dot(axis)
            );
        }
    }

    #[test]
    fn mist_texture_edges_are_transparent_and_center_opaque() {
        let img = build_soft_radial_mist_texture(64);
        let data = img.data.as_ref().unwrap();
        // Corner pixel (0, 0) must have alpha == 0
        assert_eq!(data[3], 0);
        // Center pixel (32, 32) must have high alpha
        let c_idx = (32 * 64 + 32) * 4;
        assert!(data[c_idx + 3] > 200);
    }
}
