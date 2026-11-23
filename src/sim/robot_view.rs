use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::core::types::{LinkTransform, NozzleCap, RackSlotPose};
use crate::sim::state::{AppConfig, SimStepSet, SprayJobRes};

pub fn plugin(app: &mut App) {
    app.init_resource::<HeldCanSlot>()
        .init_resource::<SlotCapVisualState>()
        .init_resource::<StudioRackCache>()
        .add_systems(Startup, spawn_robot_and_cans)
        .add_systems(Update, update_dynamic_dress_pack.in_set(SimStepSet::CanvasAndFx));
}

/// Identifies a moving link entity driven by [`KinematicChain::fk_links`].
#[derive(Component, Debug, Clone, Copy)]
pub struct RobotLink {
    pub index: usize,
}

/// Identifies a parallel gripper jaw child of Link 8 (`flange_roll`; `side`: `-1.0` left, `+1.0` right).
#[derive(Component, Debug, Clone, Copy)]
pub struct GripperJaw {
    pub side: f32,
}

/// Identifies the servo trigger finger child of Link 8 (`flange_roll`) that depresses the aerosol nozzle cap.
#[derive(Component, Debug, Clone, Copy)]
pub struct TriggerFinger;

/// Identifies a spawned aerosol spray can corresponding to `palette.cans[slot]`.
#[derive(Component, Debug, Clone, Copy)]
pub struct CanEntity {
    pub slot: usize,
}

/// Identifies a switchable nozzle cap visual group attached to a [`CanEntity`].
#[derive(Component, Debug, Clone, Copy)]
pub struct CanCapVisual {
    pub slot: usize,
    pub cap: NozzleCap,
}

/// Obstacle-clearance rule pushing the interior of a flexing [`DynamicJointHose`]
/// cleanly outside joint drums, shafts, and counterweight housings.
#[derive(Component, Debug, Clone, Copy)]
pub enum HoseClearance {
    None,
    RadialAxis {
        link: usize,
        axis: Vec3,
        min_radius: f32,
        bias_local: Vec3,
    },
    ElbowCounterweight {
        link4: usize,
    },
}

/// Flexing spline hose/conduit mesh continuously updated to bridge adjacent [`RobotLink`]s.
#[derive(Component, Debug, Clone)]
pub struct DynamicJointHose {
    pub link_a: usize,
    pub pos_a: Vec3,
    pub tan_a: Vec3,
    pub link_b: usize,
    pub pos_b: Vec3,
    pub tan_b: Vec3,
    pub radius: f32,
    pub rings: usize,
    pub sides: usize,
    pub slack: f32,
    pub uv_v_scale: f32,
    pub clearance: HoseClearance,
    pub mesh: Handle<Mesh>,
}

/// Rolling floor energy drag-chain inside the linear rail tray that tracks `rail_x` (`q[0]`).
#[derive(Component, Debug, Clone)]
pub struct DynamicDragChain {
    pub rail_z: f32,
    pub tray_x_anchor: f32,
    pub chain_len: f32,
    pub bend_radius: f32,
    pub samples: usize,
    pub mesh: Handle<Mesh>,
}

/// Tracks which rack slot's spray can is currently held by the gripper.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct HeldCanSlot(pub Option<usize>);

/// Tracks the currently mounted nozzle cap visual state for each rack slot.
#[derive(Resource, Debug, Default, Clone)]
pub struct SlotCapVisualState(pub Vec<NozzleCap>);

/// Cached rack slot poses computed from [`StudioConfig`] at startup.
#[derive(Resource, Debug, Default, Clone)]
pub struct StudioRackCache {
    pub slots: Vec<RackSlotPose>,
}

const TRIGGER_PIVOT_LOCAL: Vec3 = Vec3::new(0.0, 0.114, -0.006);

pub fn jaw_local_translation(side: f32, gripper_open: f32) -> Vec3 {
    let g = gripper_open.clamp(0.0, 1.0);
    // Can radius is 0.033m, jaw thickness is 0.012m -> closed center at ±0.039m, open at ±0.057m
    Vec3::new(side * (0.039 + 0.018 * g), -0.008, -0.025)
}

pub fn trigger_local_transform(trigger_depressed: f32) -> Transform {
    let t = trigger_depressed.clamp(0.0, 1.0);
    // Negative rotation around +X swings the -Z finger tip downward (-Y) onto the nozzle cap.
    Transform::from_translation(TRIGGER_PIVOT_LOCAL)
        .with_rotation(Quat::from_rotation_x(-0.24 * t))
}

struct RobotMaterials {
    cast_iron: Handle<StandardMaterial>,
    anodized_dark: Handle<StandardMaterial>,
    safety_orange: Handle<StandardMaterial>,
    safety_orange_distal: Handle<StandardMaterial>,
    graphite: Handle<StandardMaterial>,
    aluminum: Handle<StandardMaterial>,
    rubber: Handle<StandardMaterial>,
    bolt_steel: Handle<StandardMaterial>,
    conduit_straight: Handle<StandardMaterial>,
    air_hose: Handle<StandardMaterial>,
    brass: Handle<StandardMaterial>,
    unit07_plate: Handle<StandardMaterial>,
    hazard_chevron: Handle<StandardMaterial>,
    encoder_dial: Handle<StandardMaterial>,
    degree_scale: Handle<StandardMaterial>,
    led_green: Handle<StandardMaterial>,
    led_amber: Handle<StandardMaterial>,
}

struct RobotSharedMeshes {
    bolt_m8: Handle<Mesh>,
    bolt_m5: Handle<Mesh>,
    bolt_washer_m8: Handle<Mesh>,
    led_bezel: Handle<Mesh>,
    led_dot: Handle<Mesh>,
    clamp_saddle: Handle<Mesh>,
    air_fitting: Handle<Mesh>,
}

fn spawn_robot_and_cans(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    app_config: Res<AppConfig>,
    spray_job: Res<SprayJobRes>,
    mut held_can: ResMut<HeldCanSlot>,
    mut slot_caps: ResMut<SlotCapVisualState>,
    mut rack_cache: ResMut<StudioRackCache>,
) {
    let studio = &app_config.studio;
    let max_slots = app_config
        .cli
        .max_cans
        .max(spray_job.palette.cans.iter().map(|c| c.slot_index + 1).max().unwrap_or(0));
    rack_cache.slots = studio.rack_slots(max_slots);

    let frame = &spray_job.current_frame;
    held_can.0 = frame.held_can_slot;
    slot_caps.0 = vec![NozzleCap::Fat; max_slots];
    if let Some(slot) = frame.held_can_slot
        && let Some(entry) = slot_caps.0.get_mut(slot)
    {
        *entry = frame.active_cap;
    }

    let link_xforms = spray_job.chain.fk_links(&frame.joints);
    let grasp_xform = spray_job.chain.fk_grasp(&frame.joints);

    let mats = build_robot_materials(&mut materials, &mut images);
    let shared = RobotSharedMeshes {
        bolt_m8: meshes.add(with_tangents(Cylinder::new(0.0058, 0.0065).mesh().resolution(12).build())),
        bolt_m5: meshes.add(with_tangents(Cylinder::new(0.0034, 0.0045).mesh().resolution(10).build())),
        bolt_washer_m8: meshes.add(with_tangents(Cylinder::new(0.0080, 0.0020).mesh().resolution(12).build())),
        led_bezel: meshes.add(with_tangents(Cylinder::new(0.0055, 0.004).mesh().resolution(12).build())),
        led_dot: meshes.add(Cylinder::new(0.0032, 0.005).mesh().resolution(12).build()),
        clamp_saddle: meshes.add(with_tangents(Cuboid::new(0.056, 0.016, 0.022).mesh().build())),
        air_fitting: meshes.add(with_tangents(Cylinder::new(0.0048, 0.013).mesh().resolution(12).build())),
    };

    let link_tf = |idx: usize| -> Transform {
        let lt = link_xforms[idx];
        Transform {
            translation: lt.pos,
            rotation: lt.rot,
            scale: Vec3::ONE,
        }
    };

    // Link 0: rail_x cast-iron linear carriage + bearing runners + flanged base ring + hazard bumpers
    let bumper_plate_x = meshes.add(make_decal_quad_mesh(0.26, 0.036));
    let turret_ring_bolts = bolt_ring_transforms(0.106, 8, Vec3::Y, Vec3::new(0.0, 0.089, 0.0));
    commands
        .spawn((RobotLink { index: 0 }, link_tf(0), Visibility::default()))
        .with_children(|p| {
            // Main carriage saddle
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.38, 0.056, 0.30).mesh().build()))),
                MeshMaterial3d(mats.cast_iron.clone()),
                Transform::from_xyz(0.0, -0.022, 0.0),
            ));
            // 4 linear bearing runner blocks riding the twin guideways (Z = ±0.095)
            let runner_mesh = meshes.add(with_tangents(Cuboid::new(0.095, 0.036, 0.048).mesh().build()));
            for &(rx, rz) in &[(-0.13, -0.095), (0.13, -0.095), (-0.13, 0.095), (0.13, 0.095)] {
                p.spawn((
                    Mesh3d(runner_mesh.clone()),
                    MeshMaterial3d(mats.anodized_dark.clone()),
                    Transform::from_xyz(rx, -0.052, rz),
                ));
            }
            // Top mounting plate & flanged turret ring
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.28, 0.034, 0.28).mesh().build()))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_xyz(0.0, 0.022, 0.0),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.122, 0.052).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.cast_iron.clone()),
                Transform::from_xyz(0.0, 0.062, 0.0),
            ));
            // 4 mounting plate corner bolts + 8 turret flange bolts
            for &(bx, bz) in &[(-0.118, -0.118), (0.118, -0.118), (-0.118, 0.118), (0.118, 0.118)] {
                spawn_bolt(p, &shared.bolt_m8, &shared.bolt_washer_m8, &mats.bolt_steel, Transform::from_xyz(bx, 0.041, bz));
            }
            for tf in turret_ring_bolts {
                spawn_bolt(p, &shared.bolt_m8, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
            // Pinch/crush hazard chevron bumper plates on carriage ±X ends
            p.spawn((
                Mesh3d(bumper_plate_x.clone()),
                MeshMaterial3d(mats.hazard_chevron.clone()),
                Transform::from_xyz(0.1912, -0.022, 0.0).with_rotation(Quat::from_rotation_y(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(bumper_plate_x),
                MeshMaterial3d(mats.hazard_chevron.clone()),
                Transform::from_xyz(-0.1912, -0.022, 0.0).with_rotation(Quat::from_rotation_y(-FRAC_PI_2)),
            ));
            // Front/canvas-side servo drive pack
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.11, 0.072, 0.09).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, -0.012, -0.17),
            ));
            // Rear (+Z) pneumatic/cable junction pack, dress-pack gland & status LEDs (visible to cameras)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.128, 0.072, 0.078).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, -0.008, 0.175),
            ));
            spawn_status_leds(
                p,
                &shared,
                &mats,
                Transform::from_xyz(0.036, 0.010, 0.214).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                Transform::from_xyz(0.016, 0.010, 0.214).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
            // Compact anodized dress-pack exit gland collars meeting dynamic Link 0 -> Link 1 hoses
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0195, 0.014).mesh().resolution(16).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.016, 0.028, 0.168),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0105, 0.014).mesh().resolution(12).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(-0.014, 0.028, 0.168),
            ));
        });

    // Link 1: base_yaw rotating riser pedestal & shoulder clevis yoke
    // Frame is at shoulder_dy * 0.5 = 0.485m above Link 0 and 0.485m below Link 2.
    let clevis_chevron = meshes.add(make_decal_quad_mesh(0.132, 0.030));
    let base_flange_bolts = bolt_ring_transforms(0.112, 12, Vec3::Y, Vec3::new(0.0, -0.358, 0.0));
    commands
        .spawn((RobotLink { index: 1 }, link_tf(1), Visibility::default()))
        .with_children(|p| {
            // Lower bolted flange meeting carriage turret ring
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.126, 0.044).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, -0.382, 0.0),
            ));
            // Base degree scale collar
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.1275, 0.022).mesh().resolution(40).build()))),
                MeshMaterial3d(mats.degree_scale.clone()),
                Transform::from_xyz(0.0, -0.384, 0.0),
            ));
            for tf in base_flange_bolts {
                spawn_bolt(p, &shared.bolt_m8, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
            // Tapered graphite riser column
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    ConicalFrustum {
                        radius_top: 0.096,
                        radius_bottom: 0.118,
                        height: 0.70,
                    }
                    .mesh()
                    .resolution(32)
                    .build(),
                ))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_xyz(0.0, -0.02, 0.0),
            ));
            // Safety-orange turret shoulder collar
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.108, 0.11).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.safety_orange.clone()),
                Transform::from_xyz(0.0, 0.345, 0.0),
            ));
            // Twin clevis yoke cheeks up to shoulder pivot (y_local = 0.485)
            let cheek_mesh = meshes.add(with_tangents(Cuboid::new(0.042, 0.22, 0.155).mesh().build()));
            p.spawn((
                Mesh3d(cheek_mesh.clone()),
                MeshMaterial3d(mats.cast_iron.clone()),
                Transform::from_xyz(-0.092, 0.435, 0.0),
            ));
            p.spawn((
                Mesh3d(cheek_mesh),
                MeshMaterial3d(mats.cast_iron.clone()),
                Transform::from_xyz(0.092, 0.435, 0.0),
            ));
            // Pinch/crush hazard chevron bands on outer shoulder clevis cheeks
            p.spawn((
                Mesh3d(clevis_chevron.clone()),
                MeshMaterial3d(mats.hazard_chevron.clone()),
                Transform::from_xyz(0.1142, 0.365, 0.0).with_rotation(Quat::from_rotation_y(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(clevis_chevron),
                MeshMaterial3d(mats.hazard_chevron.clone()),
                Transform::from_xyz(-0.1142, 0.365, 0.0).with_rotation(Quat::from_rotation_y(-FRAC_PI_2)),
            ));
            // UNIT-07 machined side identification plates on Link 1 riser pedestal (±X flanks)
            let riser_boss = meshes.add(with_tangents(Cuboid::new(0.022, 0.056, 0.142).mesh().build()));
            let riser_unit07 = meshes.add(make_decal_quad_mesh(0.136, 0.051));
            p.spawn((
                Mesh3d(riser_boss.clone()),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.100, 0.145, 0.0),
            ));
            p.spawn((
                Mesh3d(riser_unit07.clone()),
                MeshMaterial3d(mats.unit07_plate.clone()),
                Transform::from_xyz(0.1122, 0.145, 0.0).with_rotation(Quat::from_rotation_y(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(riser_boss),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(-0.100, 0.145, 0.0),
            ));
            p.spawn((
                Mesh3d(riser_unit07),
                MeshMaterial3d(mats.unit07_plate.clone()),
                Transform::from_xyz(-0.1122, 0.145, 0.0).with_rotation(Quat::from_rotation_y(-FRAC_PI_2)),
            ));
            // Rear (+Z) riser dress-pack clamps, corrugated conduit & pneumatic air hose (y = -0.365..+0.345)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0155, 0.71).mesh().resolution(16).build()))),
                MeshMaterial3d(mats.conduit_straight.clone()),
                Transform::from_xyz(0.016, -0.010, 0.136),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0068, 0.71).mesh().resolution(12).build()))),
                MeshMaterial3d(mats.air_hose.clone()),
                Transform::from_xyz(-0.014, -0.010, 0.136),
            ));
            for &cy in &[-0.32_f32, -0.10, 0.12, 0.30] {
                p.spawn((
                    Mesh3d(shared.clamp_saddle.clone()),
                    MeshMaterial3d(mats.aluminum.clone()),
                    Transform::from_xyz(0.0, cy, 0.122),
                ));
            }
        });

    // Link 2: shoulder_pitch lower half of upper arm (extends +Y by L2 * 0.5 = 0.34m to upper_arm_roll)
    let half_l2 = studio.upper_arm_l2 * 0.5;
    let shoulder_cap_mesh = meshes.add(with_tangents(Cylinder::new(0.062, 0.004).mesh().resolution(32).build()));
    let shoulder_bolts_pos = bolt_ring_transforms(0.056, 8, Vec3::X, Vec3::new(0.133, 0.0, 0.0));
    let shoulder_bolts_neg = bolt_ring_transforms(0.056, 8, Vec3::NEG_X, Vec3::new(-0.133, 0.0, 0.0));
    commands
        .spawn((RobotLink { index: 2 }, link_tf(2), Visibility::default()))
        .with_children(|p| {
            // Shoulder joint drum & anodized motor caps
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.082, 0.235).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.068, 0.265).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            // Machined degree scale band around shoulder drum
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0835, 0.022).mesh().resolution(36).build()))),
                MeshMaterial3d(mats.degree_scale.clone()),
                Transform::from_xyz(0.086, 0.0, 0.0).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            // Machined encoder dial caps & bolt rings on both shoulder motor ends
            p.spawn((
                Mesh3d(shoulder_cap_mesh.clone()),
                MeshMaterial3d(mats.encoder_dial.clone()),
                Transform::from_xyz(0.134, 0.0, 0.0).with_rotation(Quat::from_rotation_z(-FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(shoulder_cap_mesh),
                MeshMaterial3d(mats.encoder_dial.clone()),
                Transform::from_xyz(-0.134, 0.0, 0.0).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            for tf in shoulder_bolts_pos.into_iter().chain(shoulder_bolts_neg) {
                spawn_bolt(p, &shared.bolt_m5, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
            // Status LED housing on shoulder rear-right shoulder bezel
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.024, 0.026, 0.036).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.072, 0.052, 0.056),
            ));
            spawn_status_leds(
                p,
                &shared,
                &mats,
                Transform::from_xyz(0.084, 0.058, 0.062).with_rotation(Quat::from_rotation_z(-FRAC_PI_2)),
                Transform::from_xyz(0.084, 0.045, 0.062).with_rotation(Quat::from_rotation_z(-FRAC_PI_2)),
            );
            // Tapered safety-orange lower upper-arm beam (y = 0 .. half_l2)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    ConicalFrustum {
                        radius_top: 0.062,
                        radius_bottom: 0.072,
                        height: (half_l2 - 0.04).max(0.05),
                    }
                    .mesh()
                    .resolution(32)
                    .build(),
                ))),
                MeshMaterial3d(mats.safety_orange.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5, 0.0),
            ));
            // Structural graphite spine rib
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cuboid::new(0.046, (half_l2 - 0.08).max(0.05), 0.136).mesh().build(),
                ))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5, 0.012),
            ));
            // Machined dark-anodized & aluminum rotary swivel ring terminating at y = half_l2 (0.34m)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.066, 0.022).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, half_l2 - 0.017, 0.0),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.069, 0.006).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, half_l2 - 0.003, 0.0),
            ));
            // Rear (+Z) dress-pack conduit & air hose clamped cleanly along Link 2 spine (y = 0.075..0.275)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cylinder::new(0.014, 0.20).mesh().resolution(16).build(),
                ))),
                MeshMaterial3d(mats.conduit_straight.clone()),
                Transform::from_xyz(0.016, 0.175, 0.100),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cylinder::new(0.0062, 0.20).mesh().resolution(12).build(),
                ))),
                MeshMaterial3d(mats.air_hose.clone()),
                Transform::from_xyz(-0.012, 0.175, 0.100),
            ));
            for &cy in &[0.095_f32, 0.245] {
                p.spawn((
                    Mesh3d(shared.clamp_saddle.clone()),
                    MeshMaterial3d(mats.aluminum.clone()),
                    Transform::from_xyz(0.0, cy, 0.088),
                ));
            }
        });

    // Link 3: upper_arm_roll rotary swivel joint at mid-upper-arm (+Y axis, extends +Y by L2 * 0.5 = 0.34m to elbow_pitch)
    let swivel_bolts = bolt_ring_transforms(0.059, 8, Vec3::Y, Vec3::new(0.0, 0.030, 0.0));
    commands
        .spawn((RobotLink { index: 3 }, link_tf(3), Visibility::default()))
        .with_children(|p| {
            // Sleek rotary bearing collar at y = 0
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.068, 0.008).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.004, 0.0),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.066, 0.020).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, 0.018, 0.0),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0672, 0.014).mesh().resolution(36).build()))),
                MeshMaterial3d(mats.degree_scale.clone()),
                Transform::from_xyz(0.0, 0.018, 0.0),
            ));
            for tf in swivel_bolts {
                spawn_bolt(p, &shared.bolt_m5, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
            // Tapered safety-orange upper segment (y = 0 .. half_l2)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    ConicalFrustum {
                        radius_top: 0.054,
                        radius_bottom: 0.062,
                        height: (half_l2 - 0.045).max(0.05),
                    }
                    .mesh()
                    .resolution(32)
                    .build(),
                ))),
                MeshMaterial3d(mats.safety_orange.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5 + 0.005, 0.0),
            ));
            // Structural graphite rib up to elbow drum
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cuboid::new(0.044, (half_l2 - 0.09).max(0.05), 0.122).mesh().build(),
                ))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_xyz(0.0, half_l2 * 0.5 + 0.005, 0.012),
            ));
            // Elbow-end anodized collar
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.060, 0.036).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, half_l2 - 0.036, 0.0),
            ));
            // Rear (+Z) upper-arm dress-pack conduit & air hose clamped cleanly along Link 3 (y = 0.085..0.265)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cylinder::new(0.013, 0.18).mesh().resolution(16).build(),
                ))),
                MeshMaterial3d(mats.conduit_straight.clone()),
                Transform::from_xyz(0.015, 0.175, 0.091),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cylinder::new(0.0058, 0.18).mesh().resolution(12).build(),
                ))),
                MeshMaterial3d(mats.air_hose.clone()),
                Transform::from_xyz(-0.012, 0.175, 0.091),
            ));
            p.spawn((
                Mesh3d(shared.clamp_saddle.clone()),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.175, 0.080),
            ));
        });

    // Link 4: elbow_pitch forearm (extends -Z by L3 = 0.60m to wrist_1)
    let l3 = studio.forearm_l3;
    let elbow_cap_mesh = meshes.add(with_tangents(Cylinder::new(0.058, 0.004).mesh().resolution(32).build()));
    let elbow_bolts_pos = bolt_ring_transforms(0.056, 8, Vec3::X, Vec3::new(0.098, 0.0, 0.0));
    let elbow_bolts_neg = bolt_ring_transforms(0.056, 8, Vec3::NEG_X, Vec3::new(-0.098, 0.0, 0.0));
    let unit07_side_quad = meshes.add(make_decal_quad_mesh(0.076, 0.0285));
    let unit07_side_backing = meshes.add(with_tangents(Cuboid::new(0.003, 0.032, 0.080).mesh().build()));
    commands
        .spawn((RobotLink { index: 4 }, link_tf(4), Visibility::default()))
        .with_children(|p| {
            // Elbow joint drum & anodized caps
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.068, 0.195).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.0695, 0.020).mesh().resolution(36).build()))),
                MeshMaterial3d(mats.degree_scale.clone()),
                Transform::from_xyz(0.072, 0.0, 0.0).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(elbow_cap_mesh.clone()),
                MeshMaterial3d(mats.encoder_dial.clone()),
                Transform::from_xyz(0.0985, 0.0, 0.0).with_rotation(Quat::from_rotation_z(-FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(elbow_cap_mesh),
                MeshMaterial3d(mats.encoder_dial.clone()),
                Transform::from_xyz(-0.0985, 0.0, 0.0).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            for tf in elbow_bolts_pos.into_iter().chain(elbow_bolts_neg) {
                spawn_bolt(p, &shared.bolt_m5, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
            // Rear servo housing counterweight
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.108, 0.118, 0.145).mesh().build()))),
                MeshMaterial3d(mats.cast_iron.clone()),
                Transform::from_xyz(0.0, 0.012, 0.078),
            ));
            // UNIT-07 stencil identification plates mounted flush on counterweight (±X behind drum + +Z rear face)
            p.spawn((
                Mesh3d(unit07_side_backing.clone()),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.055, 0.024, 0.109),
            ));
            p.spawn((
                Mesh3d(unit07_side_quad.clone()),
                MeshMaterial3d(mats.unit07_plate.clone()),
                Transform::from_xyz(0.057, 0.024, 0.109).with_rotation(Quat::from_rotation_y(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(unit07_side_backing),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(-0.055, 0.024, 0.109),
            ));
            p.spawn((
                Mesh3d(unit07_side_quad),
                MeshMaterial3d(mats.unit07_plate.clone()),
                Transform::from_xyz(-0.057, 0.024, 0.109).with_rotation(Quat::from_rotation_y(-FRAC_PI_2)),
            ));
            // Compact anodized rear cable-guide saddle bar framing the elbow dress-pack
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.068, 0.024, 0.012).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, 0.026, 0.156),
            ));
            // Safety-orange tapered forearm tube (-Z direction: top/v=1 points along -Z toward wrist)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    ConicalFrustum {
                        radius_top: 0.042,
                        radius_bottom: 0.058,
                        height: l3 - 0.04,
                    }
                    .mesh()
                    .resolution(32)
                    .build(),
                ))),
                MeshMaterial3d(mats.safety_orange_distal.clone()),
                Transform::from_xyz(0.0, 0.0, -l3 * 0.5).with_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
            ));
            // Top graphite cable dress-pack spine rail
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.028, 0.024, l3 - 0.12).mesh().build()))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_xyz(0.0, 0.058, -l3 * 0.5),
            ));
            // Forearm corrugated conduit & pneumatic air hose clamped along top (+Y) spine (z = -0.11..-0.49)
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cylinder::new(0.012, (l3 - 0.22).max(0.10)).mesh().resolution(16).build(),
                ))),
                MeshMaterial3d(mats.conduit_straight.clone()),
                Transform::from_xyz(0.012, 0.080, -l3 * 0.5)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(
                    Cylinder::new(0.0055, (l3 - 0.22).max(0.10)).mesh().resolution(12).build(),
                ))),
                MeshMaterial3d(mats.air_hose.clone()),
                Transform::from_xyz(-0.011, 0.078, -l3 * 0.5)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
            // 3 machined aluminum P-clamp brackets along forearm dress-pack
            let forearm_clamp = meshes.add(with_tangents(Cuboid::new(0.050, 0.020, 0.022).mesh().build()));
            for &cz in &[-0.14_f32, -0.30, -0.46] {
                p.spawn((
                    Mesh3d(forearm_clamp.clone()),
                    MeshMaterial3d(mats.aluminum.clone()),
                    Transform::from_xyz(0.0, 0.076, cz),
                ));
            }
            // Wrist-base anodized collar
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.046, 0.036).mesh().resolution(32).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, 0.0, -l3 + 0.02).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
        });

    // Link 5: wrist_1 machined aluminum roll housing (extends -Z by 0.065m to wrist_2)
    let wrist1_bolts = bolt_ring_transforms(0.035, 6, Vec3::NEG_Z, Vec3::new(0.0, 0.0, -0.047));
    commands
        .spawn((RobotLink { index: 5 }, link_tf(5), Visibility::default()))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.042, 0.046).mesh().resolution(28).build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.0, -0.023).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
            let cheek_mesh = meshes.add(with_tangents(Cuboid::new(0.016, 0.058, 0.044).mesh().build()));
            p.spawn((
                Mesh3d(cheek_mesh.clone()),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(-0.040, 0.0, -0.054),
            ));
            p.spawn((
                Mesh3d(cheek_mesh),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.040, 0.0, -0.054),
            ));
            // Wrist dress-pack saddle clamp & machined brass pneumatic Y-splitter manifold on +Y
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.036, 0.012, 0.018).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, 0.046, -0.022),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.028, 0.012, 0.014).mesh().build()))),
                MeshMaterial3d(mats.brass.clone()),
                Transform::from_xyz(0.0, 0.052, -0.028),
            ));
            for tf in wrist1_bolts {
                spawn_bolt(p, &shared.bolt_m5, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
        });

    // Link 6: wrist_2 pitch clevis (extends -Z by 0.030m to wrist_yaw)
    commands
        .spawn((RobotLink { index: 6 }, link_tf(6), Visibility::default()))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.028, 0.098).mesh().resolution(24).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.048, 0.046, 0.024).mesh().build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.0, -0.012),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.023, 0.008).mesh().resolution(24).build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.0, -0.026).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
        });

    // Link 7: wrist_yaw vertical rotary drum + machined aluminum U-yoke cheeks (+Y axis, extends -Z by 0.020m to flange_roll)
    commands
        .spawn((RobotLink { index: 7 }, link_tf(7), Visibility::default()))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.024, 0.064).mesh().resolution(24).build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::IDENTITY,
            ));
            let yoke_mesh = meshes.add(with_tangents(Cuboid::new(0.009, 0.056, 0.024).mesh().build()));
            p.spawn((
                Mesh3d(yoke_mesh.clone()),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(-0.026, 0.0, -0.008),
            ));
            p.spawn((
                Mesh3d(yoke_mesh),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.026, 0.0, -0.008),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.028, 0.012).mesh().resolution(24).build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.0, -0.014).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
        });

    // Link 8: flange_roll tool flange + parallel pneumatic gripper + servo trigger finger
    // Grasp center (tcp_offset) is at local (0, 0, -0.025).
    let jaw_mesh = meshes.add(with_tangents(Cuboid::new(0.012, 0.044, 0.040).mesh().build()));
    let jaw_pad_mesh = meshes.add(with_tangents(Cuboid::new(0.004, 0.038, 0.034).mesh().build()));
    let wrist_flange_bolts = bolt_ring_transforms(0.033, 8, Vec3::Z, Vec3::new(0.0, 0.0, 0.001));

    commands
        .spawn((RobotLink { index: 8 }, link_tf(8), Visibility::default()))
        .with_children(|p| {
            // Tool flange disc
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cylinder::new(0.039, 0.012).mesh().resolution(28).build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.0, -0.006).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ));
            for tf in wrist_flange_bolts {
                spawn_bolt(p, &shared.bolt_m5, &shared.bolt_washer_m8, &mats.bolt_steel, tf);
            }
            // Pneumatic parallel gripper body & linear rail
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.118, 0.036, 0.018).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_xyz(0.0, 0.0, -0.012),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.112, 0.014, 0.010).mesh().build()))),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_xyz(0.0, 0.0, -0.022),
            ));
            // Brass pneumatic push-to-connect air fittings on top rear of gripper body (fed by DynamicJointHose jumpers)
            for &fx in &[-0.044_f32, 0.044] {
                p.spawn((
                    Mesh3d(shared.air_fitting.clone()),
                    MeshMaterial3d(mats.brass.clone()),
                    Transform::from_xyz(fx, 0.022, -0.010),
                ));
            }
            // Left parallel gripper jaw (-X)
            p.spawn((
                GripperJaw { side: -1.0 },
                Mesh3d(jaw_mesh.clone()),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_translation(jaw_local_translation(-1.0, frame.gripper_open)),
            ))
            .with_children(|jaw| {
                jaw.spawn((
                    Mesh3d(jaw_pad_mesh.clone()),
                    MeshMaterial3d(mats.rubber.clone()),
                    Transform::from_xyz(0.006, 0.0, 0.0),
                ));
            });
            // Right parallel gripper jaw (+X)
            p.spawn((
                GripperJaw { side: 1.0 },
                Mesh3d(jaw_mesh),
                MeshMaterial3d(mats.aluminum.clone()),
                Transform::from_translation(jaw_local_translation(1.0, frame.gripper_open)),
            ))
            .with_children(|jaw| {
                jaw.spawn((
                    Mesh3d(jaw_pad_mesh),
                    MeshMaterial3d(mats.rubber.clone()),
                    Transform::from_xyz(-0.006, 0.0, 0.0),
                ));
            });
            // Vertical trigger-servo mounting tower & micro-servo housing
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.018, 0.098, 0.014).mesh().build()))),
                MeshMaterial3d(mats.graphite.clone()),
                Transform::from_xyz(0.0, 0.065, -0.006),
            ));
            p.spawn((
                Mesh3d(meshes.add(with_tangents(Cuboid::new(0.026, 0.026, 0.022).mesh().build()))),
                MeshMaterial3d(mats.anodized_dark.clone()),
                Transform::from_translation(TRIGGER_PIVOT_LOCAL),
            ));
            // Actuating servo trigger finger pivoting down onto nozzle actuator cap
            p.spawn((
                TriggerFinger,
                trigger_local_transform(frame.trigger_depressed),
                Visibility::default(),
            ))
            .with_children(|tf_p| {
                tf_p.spawn((
                    Mesh3d(meshes.add(with_tangents(Cuboid::new(0.010, 0.009, 0.026).mesh().build()))),
                    MeshMaterial3d(mats.aluminum.clone()),
                    Transform::from_xyz(0.0, -0.004, -0.013),
                ));
                tf_p.spawn((
                    Mesh3d(meshes.add(with_tangents(Cylinder::new(0.005, 0.012).mesh().resolution(16).build()))),
                    MeshMaterial3d(mats.rubber.clone()),
                    Transform::from_xyz(0.0, -0.012, -0.022),
                ));
            });
        });

    // Dynamic articulated hoses & corrugated conduits bridging every moving joint seamlessly
    spawn_dynamic_joint_hoses(&mut commands, &mut meshes, &mats, &link_xforms);
    spawn_dynamic_drag_chain(
        &mut commands,
        &mut meshes,
        &mats,
        studio.rail_z,
        frame.joints[0] as f32,
    );

    // Shared PBR meshes & materials for aerosol spray cans (sizes & clearances preserved verbatim)
    let can_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.80, 0.84),
        metallic: 0.90,
        perceptual_roughness: 0.20,
        clearcoat: 0.35,
        clearcoat_perceptual_roughness: 0.16,
        ..default()
    });
    let white_nozzle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.94, 0.94, 0.93),
        metallic: 0.02,
        perceptual_roughness: 0.42,
        ..default()
    });
    let orifice_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.09),
        metallic: 0.0,
        perceptual_roughness: 0.65,
        ..default()
    });
    let cyan_cap_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#00E5FF").unwrap().into(),
        metallic: 0.08,
        perceptual_roughness: 0.34,
        ..default()
    });
    let needle_metal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.84, 0.87, 0.91),
        metallic: 0.94,
        perceptual_roughness: 0.16,
        ..default()
    });

    let bottom_rim_mesh = meshes.add(Cylinder::new(0.033, 0.010));
    let body_mesh = meshes.add(Cylinder::new(0.0328, 0.122).mesh().resolution(32).build());
    let crimp_ring_mesh = meshes.add(
        Torus {
            minor_radius: 0.0020,
            major_radius: 0.0315,
        }
        .mesh()
        .minor_resolution(10)
        .major_resolution(32)
        .build(),
    );
    let shoulder_mesh = meshes.add(
        ConicalFrustum {
            radius_top: 0.018,
            radius_bottom: 0.0328,
            height: 0.024,
        }
        .mesh()
        .resolution(28),
    );
    let cap_collar_mesh = meshes.add(Cylinder::new(0.020, 0.024));
    let nozzle_button_mesh = meshes.add(Cylinder::new(0.0085, 0.018));
    let nozzle_tip_mesh = meshes.add(Cylinder::new(0.0032, 0.014));
    let skinny_button_mesh = meshes.add(Cylinder::new(0.0080, 0.019));
    let skinny_collar_mesh = meshes.add(Cylinder::new(0.0096, 0.005));
    let skinny_needle_mesh = meshes.add(Cylinder::new(0.0014, 0.015));

    for can in &spray_job.palette.cans {
        let can_color = Color::srgb_u8(can.color_srgb[0], can.color_srgb[1], can.color_srgb[2]);
        let (label_albedo, label_rm) =
            build_spray_can_label_textures(can.color_srgb, can.slot_index, 256);
        let body_mat = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(images.add(label_albedo)),
            metallic_roughness_texture: Some(images.add(label_rm)),
            metallic: 1.0,
            perceptual_roughness: 1.0,
            reflectance: 0.62,
            clearcoat: 0.80,
            clearcoat_perceptual_roughness: 0.14,
            ..default()
        });
        let cap_mat = materials.add(StandardMaterial {
            base_color: can_color,
            metallic: 0.05,
            perceptual_roughness: 0.38,
            clearcoat: 0.55,
            clearcoat_perceptual_roughness: 0.22,
            ..default()
        });

        let init_tf = if frame.held_can_slot == Some(can.slot_index) {
            Transform {
                translation: grasp_xform.pos,
                rotation: grasp_xform.rot,
                scale: Vec3::ONE,
            }
        } else if let Some(slot_pose) = rack_cache.slots.get(can.slot_index) {
            Transform {
                translation: slot_pose.grasp_pos,
                rotation: slot_pose.grasp_rot,
                scale: Vec3::ONE,
            }
        } else {
            Transform::IDENTITY
        };

        let slot_cap = slot_caps.0.get(can.slot_index).copied().unwrap_or_default();
        let fat_vis = if slot_cap == NozzleCap::Fat {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        let skinny_vis = if slot_cap == NozzleCap::Skinny {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        commands.spawn((
            CanEntity {
                slot: can.slot_index,
            },
            init_tf,
            Visibility::default(),
            children![
                // Steel bottom rim (bottom at y = -0.085m resting flush on rack shelf)
                (
                    Mesh3d(bottom_rim_mesh.clone()),
                    MeshMaterial3d(can_steel_mat.clone()),
                    Transform::from_xyz(0.0, -0.080, 0.0),
                ),
                // Main colored aerosol body (-0.075..+0.047) with wraparound litho label & rolled crimp seams
                (
                    Mesh3d(body_mesh.clone()),
                    MeshMaterial3d(body_mat),
                    Transform::from_xyz(0.0, -0.014, 0.0),
                ),
                (
                    Mesh3d(crimp_ring_mesh.clone()),
                    MeshMaterial3d(can_steel_mat.clone()),
                    Transform::from_xyz(0.0, 0.047, 0.0),
                ),
                (
                    Mesh3d(crimp_ring_mesh.clone()),
                    MeshMaterial3d(can_steel_mat.clone()),
                    Transform::from_xyz(0.0, -0.075, 0.0),
                ),
                // Domed steel shoulder (+0.047..+0.071)
                (
                    Mesh3d(shoulder_mesh.clone()),
                    MeshMaterial3d(can_steel_mat.clone()),
                    Transform::from_xyz(0.0, 0.059, 0.0),
                ),
                // Colored cap rim collar (+0.068..+0.092)
                (
                    Mesh3d(cap_collar_mesh.clone()),
                    MeshMaterial3d(cap_mat),
                    Transform::from_xyz(0.0, 0.078, 0.0),
                ),
                // Switchable Fat cap visual group (white wide actuator button + dark fat orifice)
                (
                    CanCapVisual {
                        slot: can.slot_index,
                        cap: NozzleCap::Fat,
                    },
                    Transform::IDENTITY,
                    fat_vis,
                    children![
                        (
                            Mesh3d(nozzle_button_mesh.clone()),
                            MeshMaterial3d(white_nozzle_mat.clone()),
                            Transform::from_xyz(0.0, 0.089, 0.0),
                        ),
                        (
                            Mesh3d(nozzle_tip_mesh.clone()),
                            MeshMaterial3d(orifice_mat.clone()),
                            Transform::from_xyz(0.0, 0.085, -0.009)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        ),
                    ],
                ),
                // Switchable Skinny cap visual group (cyan stencil cap + fine metallic needle orifice)
                (
                    CanCapVisual {
                        slot: can.slot_index,
                        cap: NozzleCap::Skinny,
                    },
                    Transform::IDENTITY,
                    skinny_vis,
                    children![
                        (
                            Mesh3d(skinny_button_mesh.clone()),
                            MeshMaterial3d(cyan_cap_mat.clone()),
                            Transform::from_xyz(0.0, 0.0895, 0.0),
                        ),
                        (
                            Mesh3d(skinny_collar_mesh.clone()),
                            MeshMaterial3d(cyan_cap_mat.clone()),
                            Transform::from_xyz(0.0, 0.082, 0.0),
                        ),
                        (
                            Mesh3d(skinny_needle_mesh.clone()),
                            MeshMaterial3d(needle_metal_mat.clone()),
                            Transform::from_xyz(0.0, 0.085, -0.010)
                                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                        ),
                    ],
                ),
            ],
        ));
    }
}

fn spawn_bolt(
    parent: &mut ChildSpawnerCommands,
    bolt_mesh: &Handle<Mesh>,
    washer_mesh: &Handle<Mesh>,
    bolt_mat: &Handle<StandardMaterial>,
    tf: Transform,
) {
    parent.spawn((
        Mesh3d(bolt_mesh.clone()),
        MeshMaterial3d(bolt_mat.clone()),
        tf,
        children![(
            Mesh3d(washer_mesh.clone()),
            MeshMaterial3d(bolt_mat.clone()),
            Transform::from_xyz(0.0, -0.0022, 0.0).with_scale(Vec3::splat(0.75)),
        )],
    ));
}

fn spawn_status_leds(
    parent: &mut ChildSpawnerCommands,
    shared: &RobotSharedMeshes,
    mats: &RobotMaterials,
    green_tf: Transform,
    amber_tf: Transform,
) {
    for (tf, led_mat) in [
        (green_tf, mats.led_green.clone()),
        (amber_tf, mats.led_amber.clone()),
    ] {
        parent.spawn((
            Mesh3d(shared.led_bezel.clone()),
            MeshMaterial3d(mats.anodized_dark.clone()),
            tf,
            children![(
                Mesh3d(shared.led_dot.clone()),
                MeshMaterial3d(led_mat),
                Transform::from_xyz(0.0, 0.001, 0.0),
            )],
        ));
    }
}

fn bolt_ring_transforms(radius: f32, count: usize, axis: Vec3, center: Vec3) -> Vec<Transform> {
    let dir = axis.normalize();
    let rot = Quat::from_rotation_arc(Vec3::Y, dir);
    let tangent_u = rot * Vec3::X;
    let tangent_v = rot * Vec3::Z;
    (0..count)
        .map(|i| {
            let theta = TAU * (i as f32) / (count as f32);
            let pos = center + (tangent_u * theta.cos() + tangent_v * theta.sin()) * radius;
            Transform::from_translation(pos).with_rotation(rot)
        })
        .collect()
}

fn with_tangents(mesh: Mesh) -> Mesh {
    mesh.with_generated_tangents()
        .expect("robot mesh tangent generation failed")
}

/// Builds a double-UV-aligned `+Z`-facing quad mesh (`UV (0,0)` at top-left, `(1,1)` at bottom-right)
/// with generated tangents for crisp un-mirrored stencil plates and hazard chevrons.
fn make_decal_quad_mesh(width: f32, height: f32) -> Mesh {
    let hw = width * 0.5;
    let hh = height * 0.5;
    let positions = vec![
        [-hw, hh, 0.0],  // top-left
        [hw, hh, 0.0],   // top-right
        [hw, -hh, 0.0],  // bottom-right
        [-hw, -hh, 0.0], // bottom-left
    ];
    let normals = vec![[0.0, 0.0, 1.0]; 4];
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = Indices::U32(vec![0, 3, 2, 2, 1, 0]);
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(indices);
    with_tangents(mesh)
}

fn add_pbr_material_clearcoat(
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
    (albedo, rough_metal, normal): (Image, Image, Image),
    reflectance: f32,
    clearcoat: f32,
    clearcoat_perceptual_roughness: f32,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(albedo)),
        metallic_roughness_texture: Some(images.add(rough_metal)),
        normal_map_texture: Some(images.add(normal)),
        metallic: 1.0,
        perceptual_roughness: 1.0,
        reflectance,
        clearcoat,
        clearcoat_perceptual_roughness,
        ..default()
    })
}

fn add_pbr_material(
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
    textures: (Image, Image, Image),
    reflectance: f32,
) -> Handle<StandardMaterial> {
    add_pbr_material_clearcoat(materials, images, textures, reflectance, 0.0, 0.5)
}

fn build_robot_materials(
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
) -> RobotMaterials {
    let cast_iron = add_pbr_material(materials, images, build_cast_iron_textures(256), 0.5);
    let anodized_dark = add_pbr_material_clearcoat(
        materials,
        images,
        build_anodized_dark_textures(256),
        0.55,
        0.35,
        0.22,
    );
    let safety_orange = add_pbr_material_clearcoat(
        materials,
        images,
        build_safety_orange_textures(256, 0.0),
        0.52,
        0.65,
        0.18,
    );
    let safety_orange_distal = add_pbr_material_clearcoat(
        materials,
        images,
        build_safety_orange_textures(256, 1.0),
        0.52,
        0.65,
        0.18,
    );
    let graphite = add_pbr_material(materials, images, build_graphite_textures(256), 0.5);
    let aluminum = add_pbr_material(materials, images, build_brushed_aluminum_textures(256), 0.65);
    let rubber = add_pbr_material(materials, images, build_rubber_grip_textures(256), 0.35);
    let conduit_straight = add_pbr_material(materials, images, build_conduit_textures(256, true), 0.4);

    let bolt_steel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.38, 0.41, 0.46),
        metallic: 0.90,
        perceptual_roughness: 0.25,
        ..default()
    });
    let air_hose = materials.add(StandardMaterial {
        base_color: Srgba::hex("#1C6C8C").unwrap().into(),
        metallic: 0.06,
        perceptual_roughness: 0.42,
        ..default()
    });
    let brass = materials.add(StandardMaterial {
        base_color: Color::srgb(0.80, 0.64, 0.28),
        metallic: 0.90,
        perceptual_roughness: 0.24,
        ..default()
    });
    let unit07_plate = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(build_unit07_stencil_texture(512, 192))),
        metallic: 0.32,
        perceptual_roughness: 0.40,
        ..default()
    });
    let hazard_chevron = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(build_hazard_chevron_texture(256, 64))),
        metallic: 0.08,
        perceptual_roughness: 0.46,
        ..default()
    });
    let encoder_dial = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(build_encoder_dial_texture(256))),
        metallic: 0.72,
        perceptual_roughness: 0.30,
        ..default()
    });
    let degree_scale = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(build_degree_scale_collar_texture(512, 64))),
        metallic: 0.75,
        perceptual_roughness: 0.28,
        ..default()
    });
    let led_green = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.95, 0.38),
        emissive: LinearRgba::new(0.22, 1.9, 0.48, 1.0),
        ..default()
    });
    let led_amber = materials.add(StandardMaterial {
        base_color: Color::srgb(0.98, 0.68, 0.14),
        emissive: LinearRgba::new(1.85, 0.90, 0.12, 1.0),
        ..default()
    });

    RobotMaterials {
        cast_iron,
        anodized_dark,
        safety_orange,
        safety_orange_distal,
        graphite,
        aluminum,
        rubber,
        bolt_steel,
        conduit_straight,
        air_hose,
        brass,
        unit07_plate,
        hazard_chevron,
        encoder_dial,
        degree_scale,
        led_green,
        led_amber,
    }
}

#[inline]
fn hash2(x: i32, y: i32, seed: u32) -> f32 {
    let mut n = (x.wrapping_mul(374761393))
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed as i32);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    ((n ^ (n >> 16)) as u32) as f32 / (u32::MAX as f32)
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let a = hash2(xi, yi, seed);
    let b = hash2(xi + 1, yi, seed);
    let c = hash2(xi, yi + 1, seed);
    let d = hash2(xi + 1, yi + 1, seed);
    (a + (b - a) * u) + ((c - a) + (a - b - c + d) * u) * v
}

fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for oct in 0..4 {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(oct * 101));
        freq *= 2.07;
        amp *= 0.5;
    }
    sum
}

fn create_rgba8_image(width: u32, height: u32, data: Vec<u8>, format: TextureFormat) -> Image {
    let mut img = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.sampler = ImageSampler::linear();
    img
}

fn create_rgba8_repeating_image(
    width: u32,
    height: u32,
    data: Vec<u8>,
    format: TextureFormat,
) -> Image {
    let mut img = create_rgba8_image(width, height, data, format);
    let mut desc = ImageSamplerDescriptor::linear();
    desc.set_address_mode(ImageAddressMode::Repeat);
    img.sampler = ImageSampler::Descriptor(desc);
    img
}

fn normal_map_from_heights(heights: &[f32], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    let w = width as i32;
    let h = height as i32;
    for y in 0..h {
        for x in 0..w {
            let xm = (x - 1).rem_euclid(w) as usize;
            let xp = (x + 1).rem_euclid(w) as usize;
            let ym = (y - 1).rem_euclid(h) as usize;
            let yp = (y + 1).rem_euclid(h) as usize;
            let row = y as usize * width as usize;
            let dx = heights[row + xp] - heights[row + xm];
            let dy = heights[yp * width as usize + x as usize]
                - heights[ym * width as usize + x as usize];
            let n = Vec3::new(-dx * strength, -dy * strength, 1.0).normalize();
            out.push(((n.x * 0.5 + 0.5) * 255.0).round() as u8);
            out.push(((n.y * 0.5 + 0.5) * 255.0).round() as u8);
            out.push(((n.z * 0.5 + 0.5) * 255.0).round() as u8);
            out.push(255);
        }
    }
    out
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Cast iron: sand-cast pebbled micro-normal + graphite sheen variation (`roughness ~0.42..0.58`).
fn build_cast_iron_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let macro_grain = fbm(u * 14.0, v * 14.0, 101);
            let pebble = fbm(u * 62.0, v * 62.0, 137);
            let micro_pit = hash2(x as i32, y as i32, 173);

            let crest = (pebble * 0.72 + micro_pit * 0.28).clamp(0.0, 1.0);
            let luma = (0.155 + 0.038 * macro_grain + 0.036 * crest).clamp(0.11, 0.25);
            albedo.extend_from_slice(&[
                to_u8(luma * 0.96),
                to_u8(luma * 1.00),
                to_u8(luma * 1.08),
                255,
            ]);

            // Roughness in 0.42..0.58 (lower on rubbed crests, higher in sand-cast valleys)
            let roughness = (0.58 - 0.16 * crest + 0.03 * (macro_grain - 0.5)).clamp(0.42, 0.58);
            let metallic = (0.74 + 0.08 * crest).clamp(0.72, 0.84);
            rough_metal.extend_from_slice(&[0, to_u8(roughness), to_u8(metallic), 255]);

            heights[(y * size + x) as usize] = macro_grain * 0.30 + pebble * 0.55 + micro_pit * 0.15;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 2.2);
    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

/// Safety orange armor: powder-coat orange-peel normal + subtle edge wear + optional distal dried spray mist.
fn build_safety_orange_textures(size: u32, mist_strength: f32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    let orange_base = [0.910_f32, 0.365, 0.102];
    let primer_metal = [0.46_f32, 0.48, 0.52];
    let mist_palette = [
        [0.08_f32, 0.62, 0.86], // cyan
        [0.86, 0.24, 0.52],     // magenta
        [0.92, 0.74, 0.16],     // yellow-gold
        [0.14, 0.16, 0.20],     // charcoal black
        [0.92, 0.92, 0.90],     // off-white
    ];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            let orange_peel = fbm(u * 44.0, v * 44.0, 211);
            let micro_coat = fbm(u * 96.0, v * 96.0, 239);
            let panel_var = fbm(u * 6.0, v * 6.0, 257);

            // Subtle edge/collar wear near v=0 or v=1 (plus faint corner scuffs)
            let edge_dist = v.min(1.0 - v);
            let wear_noise = fbm(u * 26.0, v * 26.0, 277);
            let edge_mask = ((0.055 - edge_dist) / 0.055).clamp(0.0, 1.0);
            let wear = ((edge_mask * 1.15 + (wear_noise - 0.68) * 1.8).clamp(0.0, 1.0)
                * edge_mask)
                .clamp(0.0, 0.85);

            let mut rgb = [
                orange_base[0] * (0.96 + 0.06 * panel_var),
                orange_base[1] * (0.95 + 0.07 * panel_var),
                orange_base[2] * (0.95 + 0.08 * panel_var),
            ];
            for c in 0..3 {
                rgb[c] = rgb[c] * (1.0 - wear) + primer_metal[c] * wear;
            }

            let mut roughness = (0.27 + 0.06 * orange_peel + 0.14 * wear).clamp(0.22, 0.52);
            let metallic = (0.16 + 0.58 * wear).clamp(0.14, 0.78);
            let mut h_val = orange_peel * 0.70 + micro_coat * 0.30 - wear * 0.45;

            // Faint subtle multi-color dried spray-mist speckling on distal arm armor (v -> 1.0 = wrist)
            if mist_strength > 0.0 {
                let distal_grad = ((v - 0.25) / 0.75).clamp(0.0, 1.0).powf(1.25) * mist_strength;
                if distal_grad > 0.01 {
                    let haze = fbm(u * 15.0 + 3.2, v * 15.0 - 1.7, 307);
                    let haze_w = ((haze - 0.36) * 0.46 * distal_grad).clamp(0.0, 0.22);
                    // Subtle cool-grey/cyan overspray veil
                    rgb[0] = rgb[0] * (1.0 - haze_w) + 0.36 * haze_w;
                    rgb[1] = rgb[1] * (1.0 - haze_w) + 0.42 * haze_w;
                    rgb[2] = rgb[2] * (1.0 - haze_w) + 0.46 * haze_w;
                    roughness = (roughness + haze_w * 0.45).clamp(0.22, 0.75);

                    // Fine dried micro-droplet specks
                    let cell = 44.0_f32;
                    let gx = (u * cell).floor() as i32;
                    let gy = (v * cell).floor() as i32;
                    let fx = (u * cell).fract() - 0.5;
                    let fy = (v * cell).fract() - 0.5;
                    let spawn = hash2(gx, gy, 331);
                    if spawn > 0.68 {
                        let jx = (hash2(gx, gy, 337) - 0.5) * 0.55;
                        let jy = (hash2(gx, gy, 347) - 0.5) * 0.55;
                        let dist = ((fx - jx).powi(2) + (fy - jy).powi(2)).sqrt();
                        let rad = 0.12 + 0.18 * hash2(gx, gy, 353);
                        if dist < rad {
                            let edge = ((rad - dist) / (rad * 0.4)).clamp(0.0, 1.0);
                            let idx = (hash2(gx, gy, 359) * mist_palette.len() as f32) as usize;
                            let speck = mist_palette[idx.min(mist_palette.len() - 1)];
                            let alpha = (0.58 * edge * distal_grad).clamp(0.0, 0.62);
                            for c in 0..3 {
                                rgb[c] = rgb[c] * (1.0 - alpha) + speck[c] * alpha;
                            }
                            roughness = (roughness + 0.25 * alpha).clamp(0.22, 0.78);
                            h_val += 0.35 * alpha;
                        }
                    }
                }
            }

            albedo.extend_from_slice(&[to_u8(rgb[0]), to_u8(rgb[1]), to_u8(rgb[2]), 255]);
            rough_metal.extend_from_slice(&[0, to_u8(roughness), to_u8(metallic), 255]);
            heights[(y * size + x) as usize] = h_val;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 1.25);
    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

/// Anodized dark joints: lathe-turned concentric micro-grooves + darker recessed contrast.
fn build_anodized_dark_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let lathe = (v * 90.0 * TAU).sin() * 0.5 + 0.5;
            let band = fbm(u * 4.0, v * 36.0, 401);
            let micro = fbm(u * 28.0, v * 64.0, 419);

            let recessed = ((0.06 - v.min(1.0 - v)) / 0.06).clamp(0.0, 1.0);
            let luma = (0.088 + 0.018 * band + 0.010 * lathe - 0.022 * recessed).clamp(0.055, 0.14);
            albedo.extend_from_slice(&[
                to_u8(luma * 0.92),
                to_u8(luma * 1.00),
                to_u8(luma * 1.14),
                255,
            ]);

            let roughness = (0.28 + 0.08 * (1.0 - lathe) + 0.05 * micro + 0.08 * recessed)
                .clamp(0.24, 0.48);
            let metallic = (0.85 - 0.06 * recessed).clamp(0.78, 0.90);
            rough_metal.extend_from_slice(&[0, to_u8(roughness), to_u8(metallic), 255]);

            heights[(y * size + x) as usize] = lathe * 0.55 + micro * 0.30 + band * 0.15;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 1.35);
    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

/// Brushed aluminum: directional brushed-metal streaks + crisp studio HDR tube reflections.
fn build_brushed_aluminum_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            // Fine anisotropic streaks running primarily along U
            let streak_fine = value_noise(u * 10.0, v * 170.0, 503);
            let streak_med = fbm(u * 5.0, v * 72.0, 521);
            let micro = hash2(x as i32, y as i32, 541) * 0.08;

            let brush = (streak_fine * 0.60 + streak_med * 0.35 + micro).clamp(0.0, 1.0);
            let luma = (0.73 + 0.10 * brush).clamp(0.68, 0.87);
            albedo.extend_from_slice(&[
                to_u8(luma * 0.97),
                to_u8(luma * 0.99),
                to_u8(luma * 1.03),
                255,
            ]);

            // Low roughness (0.16..0.26) + high metallic (0.92..0.95) for crisp tube reflections
            let roughness = (0.16 + 0.10 * (1.0 - brush)).clamp(0.15, 0.28);
            let metallic = 0.93_f32;
            rough_metal.extend_from_slice(&[0, to_u8(roughness), to_u8(metallic), 255]);

            heights[(y * size + x) as usize] = brush;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 0.85);
    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

/// Structural satin graphite-steel finish for riser & spine ribs.
fn build_graphite_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let grain = fbm(u * 40.0, v * 40.0, 601);
            let blotch = fbm(u * 9.0, v * 9.0, 617);
            let luma = (0.235 + 0.055 * blotch + 0.032 * grain).clamp(0.18, 0.34);
            albedo.extend_from_slice(&[
                to_u8(luma * 0.94),
                to_u8(luma * 1.00),
                to_u8(luma * 1.12),
                255,
            ]);

            let roughness = (0.33 + 0.09 * (1.0 - grain)).clamp(0.28, 0.46);
            let metallic = (0.56 + 0.08 * grain).clamp(0.50, 0.68);
            rough_metal.extend_from_slice(&[0, to_u8(roughness), to_u8(metallic), 255]);

            heights[(y * size + x) as usize] = grain * 0.7 + blotch * 0.3;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 1.35);
    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

/// Rubber / Gripper pads: diamond-knurled & ribbed tactile pad texture.
fn build_rubber_grip_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let d1 = ((u + v) * 26.0 * PI).sin().abs();
            let d2 = ((u - v) * 26.0 * PI).sin().abs();
            let knurl = (d1 * d2).powf(0.65);
            let micro = fbm(u * 48.0, v * 48.0, 701);

            let luma = (0.048 + 0.028 * knurl + 0.010 * micro).clamp(0.035, 0.095);
            albedo.extend_from_slice(&[to_u8(luma), to_u8(luma), to_u8(luma * 1.06), 255]);

            let roughness = (0.90 - 0.14 * knurl + 0.04 * micro).clamp(0.72, 0.95);
            rough_metal.extend_from_slice(&[0, to_u8(roughness), 5, 255]);

            heights[(y * size + x) as usize] = knurl * 0.85 + micro * 0.15;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 3.0);
    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

/// Corrugated black polymer dress-pack conduit (`rings_along_v = true` for Cylinder, `false` for Torus).
fn build_conduit_textures(size: u32, rings_along_v: bool) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let axis = if rings_along_v { v } else { u };
            let ridge = ((axis * 54.0 * TAU).sin() * 0.5 + 0.5).powf(0.8);
            let micro = fbm(u * 32.0, v * 32.0, 809);

            let luma = (0.055 + 0.036 * ridge + 0.010 * micro).clamp(0.04, 0.12);
            albedo.extend_from_slice(&[to_u8(luma), to_u8(luma), to_u8(luma * 1.05), 255]);

            let roughness = (0.72 - 0.20 * ridge + 0.06 * micro).clamp(0.46, 0.82);
            rough_metal.extend_from_slice(&[0, to_u8(roughness), 14, 255]);

            heights[(y * size + x) as usize] = ridge * 0.88 + micro * 0.12;
        }
    }

    let normal = normal_map_from_heights(&heights, size, size, 3.2);
    (
        create_rgba8_repeating_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_repeating_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
        create_rgba8_repeating_image(size, size, normal, TextureFormat::Rgba8Unorm),
    )
}

#[inline]
fn dist_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let denom = ba.length_squared().max(1e-6);
    let h = (pa.dot(ba) / denom).clamp(0.0, 1.0);
    (pa - ba * h).length()
}

fn glyph_sdf(ch: char, p: Vec2) -> f32 {
    // Normalized glyph cell [0..1] x [0..1], y down.
    let segs: &[([f32; 2], [f32; 2])] = match ch {
        'U' => &[
            ([0.16, 0.04], [0.16, 0.76]),
            ([0.16, 0.76], [0.34, 0.96]),
            ([0.34, 0.96], [0.66, 0.96]),
            ([0.66, 0.96], [0.84, 0.76]),
            ([0.84, 0.04], [0.84, 0.76]),
        ],
        'N' => &[
            ([0.16, 0.04], [0.16, 0.96]),
            ([0.16, 0.06], [0.84, 0.94]),
            ([0.84, 0.04], [0.84, 0.96]),
        ],
        'I' => &[
            ([0.50, 0.04], [0.50, 0.96]),
            ([0.24, 0.04], [0.76, 0.04]),
            ([0.24, 0.96], [0.76, 0.96]),
        ],
        'T' => &[
            ([0.08, 0.04], [0.92, 0.04]),
            ([0.50, 0.04], [0.50, 0.96]),
        ],
        '-' => &[([0.20, 0.50], [0.80, 0.50])],
        '0' => &[
            ([0.16, 0.24], [0.16, 0.76]),
            ([0.84, 0.24], [0.84, 0.76]),
            ([0.16, 0.24], [0.34, 0.04]),
            ([0.34, 0.04], [0.66, 0.04]),
            ([0.66, 0.04], [0.84, 0.24]),
            ([0.16, 0.76], [0.34, 0.96]),
            ([0.34, 0.96], [0.66, 0.96]),
            ([0.66, 0.96], [0.84, 0.76]),
        ],
        '7' => &[
            ([0.12, 0.04], [0.88, 0.04]),
            ([0.88, 0.04], [0.34, 0.96]),
        ],
        _ => &[],
    };

    let mut d = f32::MAX;
    for &(a, b) in segs {
        d = d.min(dist_segment(p, Vec2::from(a), Vec2::from(b)));
    }
    // Authentic stencil bridge break on '0' and 'U' bottom seam
    if ch == '0' && (p.x - 0.50).abs() < 0.042 {
        d = d.max(0.042 - (p.x - 0.50).abs() + 0.08);
    }
    d
}

/// Crisp `512x192` industrial `UNIT-07` stencil plate texture.
fn build_unit07_stencil_texture(width: u32, height: u32) -> Image {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let w = width as f32;
    let h = height as f32;
    let chars = ['U', 'N', 'I', 'T', '-', '0', '7'];

    let text_x0 = 66.0_f32;
    let text_x1 = 486.0_f32;
    let text_y0 = 34.0_f32;
    let text_y1 = 136.0_f32;
    let slot_w = (text_x1 - text_x0) / chars.len() as f32;
    let glyph_w = slot_w * 0.76;
    let glyph_h = text_y1 - text_y0;

    for py in 0..height {
        for px in 0..width {
            let fx = px as f32 + 0.5;
            let fy = py as f32 + 0.5;
            let u = fx / w;
            let v = fy / h;

            let brush = fbm(u * 14.0, v * 55.0, 901);
            let mut rgb = [
                0.10 + 0.03 * brush,
                0.11 + 0.03 * brush,
                0.13 + 0.035 * brush,
            ];

            // Outer chamfered dark bezel & inner hairline highlight
            let ex = fx.min(w - fx);
            let ey = fy.min(h - fy);
            let edge = ex.min(ey);
            if edge < 5.0 {
                rgb = [0.05, 0.055, 0.065];
            } else if (edge - 8.0).abs() < 1.5 {
                rgb = [0.24, 0.26, 0.30];
            }

            // Left safety-orange accent stripe
            if (16.0..=44.0).contains(&fx) && (16.0..=(h - 16.0)).contains(&fy) {
                rgb = [0.90, 0.36, 0.10];
            }

            // Lower stencil spec bar & alignment ticks
            if (152.0..=168.0).contains(&fy) && (66.0..=484.0).contains(&fx) {
                let seg = ((fx - 66.0) / 24.0).fract();
                if fx < 170.0 {
                    rgb = [0.90, 0.36, 0.10];
                } else if seg < 0.62 {
                    rgb = [0.34, 0.37, 0.42];
                }
            }

            // 4 corner rivet/screw heads
            for &(rx, ry) in &[
                (18.0, 18.0),
                (w - 18.0, 18.0),
                (18.0, h - 18.0),
                (w - 18.0, h - 18.0),
            ] {
                let rd = ((fx - rx).powi(2) + (fy - ry).powi(2)).sqrt();
                if rd < 5.2 {
                    rgb = if rd < 2.2 {
                        [0.06, 0.065, 0.075]
                    } else {
                        [0.42, 0.45, 0.50]
                    };
                }
            }

            // Crisp SDF stencil glyphs for "UNIT-07"
            if fy >= text_y0 - 10.0 && fy <= text_y1 + 10.0 && fx >= text_x0 - 8.0 && fx <= text_x1 + 8.0 {
                let idx = ((fx - text_x0) / slot_w).floor() as isize;
                if idx >= 0 && (idx as usize) < chars.len() {
                    let gx0 = text_x0 + idx as f32 * slot_w + (slot_w - glyph_w) * 0.5;
                    let gx = (fx - gx0) / glyph_w;
                    let gy = (fy - text_y0) / glyph_h;
                    if (-0.15..=1.15).contains(&gx) && (-0.15..=1.15).contains(&gy) {
                        let d = glyph_sdf(chars[idx as usize], Vec2::new(gx, gy));
                        let stroke_r = 0.112_f32;
                        let aa = 0.024_f32;
                        let coverage = (1.0 - ((d - stroke_r) / aa).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                        if coverage > 0.0 {
                            let wear = fbm(u * 38.0, v * 38.0, 941);
                            let paint = [
                                0.93 - 0.05 * wear,
                                0.91 - 0.05 * wear,
                                0.87 - 0.05 * wear,
                            ];
                            for c in 0..3 {
                                rgb[c] = rgb[c] * (1.0 - coverage) + paint[c] * coverage;
                            }
                        }
                    }
                }
            }

            data.extend_from_slice(&[to_u8(rgb[0]), to_u8(rgb[1]), to_u8(rgb[2]), 255]);
        }
    }

    create_rgba8_image(width, height, data, TextureFormat::Rgba8UnormSrgb)
}

/// Yellow-and-black diagonal pinch/crush hazard chevron band texture.
fn build_hazard_chevron_texture(width: u32, height: u32) -> Image {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let yellow = [0.91_f32, 0.69, 0.08];
    let black = [0.09_f32, 0.095, 0.11];

    for py in 0..height {
        for px in 0..width {
            let u = px as f32 / width as f32;
            let v = py as f32 / height as f32;
            let scuff = fbm(u * 24.0, v * 12.0, 1013);

            // Thin black perimeter frame
            let border = v < 0.08 || v > 0.92 || u < 0.02 || u > 0.98;
            let phase = (u * 7.0 + v * 1.15).fract();
            let is_yellow = !border && phase < 0.50;
            let mut rgb = if is_yellow { yellow } else { black };

            // Subtle shop scuffing
            let sc = (scuff - 0.62).clamp(0.0, 0.35);
            for c in 0..3 {
                rgb[c] = (rgb[c] * (1.0 - sc * 0.55) + 0.14 * sc * 0.55).clamp(0.04, 0.96);
            }
            data.extend_from_slice(&[to_u8(rgb[0]), to_u8(rgb[1]), to_u8(rgb[2]), 255]);
        }
    }
    create_rgba8_image(width, height, data, TextureFormat::Rgba8UnormSrgb)
}

/// Radial machined servo encoder dial cap texture (`256x256`).
fn build_encoder_dial_texture(size: u32) -> Image {
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    let s = size as f32;

    for py in 0..size {
        for px in 0..size {
            let nx = ((px as f32 + 0.5) / s) * 2.0 - 1.0;
            let ny = ((py as f32 + 0.5) / s) * 2.0 - 1.0;
            let r = (nx * nx + ny * ny).sqrt();
            let ang = ny.atan2(nx).rem_euclid(TAU);

            // Concentric machined aluminum outer ring + dark recessed center
            let mut rgb = if r < 0.54 {
                [0.10_f32, 0.11, 0.13]
            } else if r < 0.92 {
                [0.64_f32, 0.67, 0.71]
            } else {
                [0.16_f32, 0.18, 0.21]
            };

            // Radial tick markings in r in [0.60, 0.88]
            let deg_step_30 = (ang / (TAU / 12.0)).fract();
            let deg_step_10 = (ang / (TAU / 36.0)).fract();
            let major_tick = (deg_step_30 < 0.08 || deg_step_30 > 0.92) && (0.60..=0.88).contains(&r);
            let minor_tick = (deg_step_10 < 0.10 || deg_step_10 > 0.90) && (0.72..=0.88).contains(&r);

            // Zero-datum safety-orange pointer at angle ≈ 0
            if (ang < 0.08 || ang > TAU - 0.08) && (0.56..=0.90).contains(&r) {
                rgb = [0.91, 0.36, 0.10];
            } else if major_tick || minor_tick {
                rgb = [0.09, 0.10, 0.12];
            }

            // Inner chamfer groove
            if (r - 0.54).abs() < 0.025 {
                rgb = [0.05, 0.055, 0.065];
            }

            data.extend_from_slice(&[to_u8(rgb[0]), to_u8(rgb[1]), to_u8(rgb[2]), 255]);
        }
    }
    create_rgba8_image(size, size, data, TextureFormat::Rgba8UnormSrgb)
}

/// Cylindrical wrap-around degree scale ring texture (`512x64`).
fn build_degree_scale_collar_texture(width: u32, height: u32) -> Image {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for py in 0..height {
        for px in 0..width {
            let u = (px as f32 + 0.5) / width as f32;
            let v = (py as f32 + 0.5) / height as f32;

            let mut rgb = [0.58_f32, 0.61, 0.66];
            if v < 0.12 || v > 0.88 {
                rgb = [0.11, 0.12, 0.14];
            } else {
                let p12 = (u * 12.0).fract();
                let p36 = (u * 36.0).fract();
                let p72 = (u * 72.0).fract();
                let major = (p12 < 0.055 || p12 > 0.945) && (0.16..=0.84).contains(&v);
                let med = (p36 < 0.08 || p36 > 0.92) && (0.28..=0.84).contains(&v);
                let fine = (p72 < 0.10 || p72 > 0.90) && (0.52..=0.84).contains(&v);

                if u < 0.008 || u > 0.992 {
                    rgb = [0.91, 0.36, 0.10];
                } else if major || med || fine {
                    rgb = [0.09, 0.10, 0.12];
                }
            }

            data.extend_from_slice(&[to_u8(rgb[0]), to_u8(rgb[1]), to_u8(rgb[2]), 255]);
        }
    }
    create_rgba8_image(width, height, data, TextureFormat::Rgba8UnormSrgb)
}

/// Generates a procedural 256x256 wraparound aerosol can litho label with prominent color swatch
/// bands, matte graphic belly label, barcode, warning icon, and subtle collar mist wear.
fn build_spray_can_label_textures(
    color_srgb: [u8; 3],
    slot_index: usize,
    size: u32,
) -> (Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);

    let paint_rgb = [
        color_srgb[0] as f32 / 255.0,
        color_srgb[1] as f32 / 255.0,
        color_srgb[2] as f32 / 255.0,
    ];
    let luma_paint = 0.299 * paint_rgb[0] + 0.587 * paint_rgb[1] + 0.114 * paint_rgb[2];
    let contrast_ink = if luma_paint > 0.55 {
        [0.09_f32, 0.10, 0.12]
    } else {
        [0.92_f32, 0.92, 0.90]
    };

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            let grain = fbm(u * 40.0 + slot_index as f32 * 1.7, v * 40.0, 907);
            let scuff = fbm(u * 18.0, v * 8.0, 919 + slot_index as u32);

            let mut rgb = paint_rgb;
            let mut rough = 0.24 + 0.05 * grain;
            let mut metallic = 0.08;

            // Central dark litho graphic label wrap (v in 0.24..0.78)
            if (0.24..=0.78).contains(&v) {
                if (0.24..=0.26).contains(&v) || (0.76..=0.78).contains(&v) {
                    // Thin crisp metallic aluminum foil pin-stripe borders
                    rgb = [0.74, 0.77, 0.81];
                    rough = 0.22;
                    metallic = 0.86;
                } else {
                    rgb = [
                        0.085 + 0.015 * grain,
                        0.092 + 0.015 * grain,
                        0.108 + 0.018 * grain,
                    ];
                    rough = 0.42 + 0.08 * grain;
                    metallic = 0.04;

                    // Prominent front brand color swatch block & diagonal graphic bar
                    if (0.34..=0.45).contains(&v) && ((u - 0.5).abs() < 0.24) {
                        rgb = paint_rgb;
                        rough = 0.28;
                    }
                    // Bold synthetic brand title bars on front half (u in 0.22..0.78)
                    if (0.49..=0.58).contains(&v) && ((u - 0.5).abs() < 0.26) {
                        let letter_cell = ((u * 28.0).fract() - 0.5).abs();
                        if letter_cell < 0.34 {
                            rgb = contrast_ink;
                            rough = 0.36;
                        }
                    }
                    // Technical spec subtitle line
                    if (0.62..=0.66).contains(&v) && ((u - 0.5).abs() < 0.20) {
                        let dot_cell = ((u * 52.0).fract() - 0.5).abs();
                        if dot_cell < 0.32 {
                            rgb = [0.76, 0.78, 0.82];
                        }
                    }
                    // Rear barcode + small orange GHS hazard diamond (u in 0.82..0.96 or 0.04..0.16)
                    if (0.38..=0.68).contains(&v) && (u > 0.82 && u < 0.96) {
                        let bc_x = (u * 190.0).floor() as i32;
                        let on_bar = hash2(bc_x, slot_index as i32, 953) > 0.42;
                        rgb = if on_bar {
                            [0.10, 0.11, 0.12]
                        } else {
                            [0.89, 0.89, 0.87]
                        };
                        rough = 0.52;
                    }
                    let du = (u - 0.11).abs();
                    let dv = (v - 0.53).abs() * 0.7;
                    let d_man = du + dv;
                    if (0.40..=0.66).contains(&v) && d_man < 0.065 {
                        if d_man > 0.045 {
                            rgb = [0.88, 0.24, 0.10];
                        } else {
                            rgb = [0.92, 0.91, 0.88];
                        }
                    }
                }
            }

            // Faint paint mist & rim handling scuff near top/bottom
            if v < 0.16 && scuff > 0.62 {
                let w = ((scuff - 0.62) * 1.4).clamp(0.0, 0.35);
                rgb[0] = rgb[0] * (1.0 - w) + paint_rgb[0] * w;
                rgb[1] = rgb[1] * (1.0 - w) + paint_rgb[1] * w;
                rgb[2] = rgb[2] * (1.0 - w) + paint_rgb[2] * w;
                rough = (rough + 0.18 * w).clamp(0.18, 0.80);
            }

            albedo.extend_from_slice(&[to_u8(rgb[0]), to_u8(rgb[1]), to_u8(rgb[2]), 255]);
            rough_metal.extend_from_slice(&[0, to_u8(rough), to_u8(metallic), 255]);
        }
    }

    (
        create_rgba8_image(size, size, albedo, TextureFormat::Rgba8UnormSrgb),
        create_rgba8_image(size, size, rough_metal, TextureFormat::Rgba8Unorm),
    )
}

struct DynamicHoseSpec {
    link_a: usize,
    pos_a: Vec3,
    tan_a: Vec3,
    link_b: usize,
    pos_b: Vec3,
    tan_b: Vec3,
    radius: f32,
    slack: f32,
    uv_v_scale: f32,
    is_conduit: bool,
    clearance: HoseClearance,
}

fn build_joint_hose_specs() -> [DynamicHoseSpec; 12] {
    [
        // 1-2: Link 0 carriage rear glands -> Link 1 riser bottom clamps across base_yaw
        DynamicHoseSpec {
            link_a: 0,
            pos_a: Vec3::new(0.016, 0.028, 0.168),
            tan_a: Vec3::Y,
            link_b: 1,
            pos_b: Vec3::new(0.016, -0.365, 0.136),
            tan_b: Vec3::Y,
            radius: 0.0155,
            slack: 0.24,
            uv_v_scale: 0.45,
            is_conduit: true,
            clearance: HoseClearance::RadialAxis {
                link: 1,
                axis: Vec3::Y,
                min_radius: 0.138,
                bias_local: Vec3::Z,
            },
        },
        DynamicHoseSpec {
            link_a: 0,
            pos_a: Vec3::new(-0.014, 0.028, 0.168),
            tan_a: Vec3::Y,
            link_b: 1,
            pos_b: Vec3::new(-0.014, -0.365, 0.136),
            tan_b: Vec3::Y,
            radius: 0.0068,
            slack: 0.24,
            uv_v_scale: 0.45,
            is_conduit: false,
            clearance: HoseClearance::RadialAxis {
                link: 1,
                axis: Vec3::Y,
                min_radius: 0.138,
                bias_local: Vec3::Z,
            },
        },
        // 3-4: Link 1 shoulder exit -> Link 2 lower spine across shoulder_pitch
        DynamicHoseSpec {
            link_a: 1,
            pos_a: Vec3::new(0.016, 0.345, 0.136),
            tan_a: Vec3::new(0.0, 0.88, 0.48).normalize(),
            link_b: 2,
            pos_b: Vec3::new(0.016, 0.075, 0.100),
            tan_b: Vec3::Y,
            radius: 0.014,
            slack: 0.38,
            uv_v_scale: 0.50,
            is_conduit: true,
            clearance: HoseClearance::RadialAxis {
                link: 2,
                axis: Vec3::X,
                min_radius: 0.106,
                bias_local: Vec3::Z,
            },
        },
        DynamicHoseSpec {
            link_a: 1,
            pos_a: Vec3::new(-0.014, 0.345, 0.136),
            tan_a: Vec3::new(0.0, 0.88, 0.48).normalize(),
            link_b: 2,
            pos_b: Vec3::new(-0.012, 0.075, 0.100),
            tan_b: Vec3::Y,
            radius: 0.0062,
            slack: 0.38,
            uv_v_scale: 0.50,
            is_conduit: false,
            clearance: HoseClearance::RadialAxis {
                link: 2,
                axis: Vec3::X,
                min_radius: 0.106,
                bias_local: Vec3::Z,
            },
        },
        // 5-6: Link 2 upper-arm clamp -> Link 3 lower clamp across upper_arm_roll swivel
        DynamicHoseSpec {
            link_a: 2,
            pos_a: Vec3::new(0.016, 0.275, 0.100),
            tan_a: Vec3::Y,
            link_b: 3,
            pos_b: Vec3::new(0.015, 0.085, 0.091),
            tan_b: Vec3::Y,
            radius: 0.0135,
            slack: 0.34,
            uv_v_scale: 0.45,
            is_conduit: true,
            clearance: HoseClearance::RadialAxis {
                link: 3,
                axis: Vec3::Y,
                min_radius: 0.090,
                bias_local: Vec3::Z,
            },
        },
        DynamicHoseSpec {
            link_a: 2,
            pos_a: Vec3::new(-0.012, 0.275, 0.100),
            tan_a: Vec3::Y,
            link_b: 3,
            pos_b: Vec3::new(-0.012, 0.085, 0.091),
            tan_b: Vec3::Y,
            radius: 0.0060,
            slack: 0.34,
            uv_v_scale: 0.45,
            is_conduit: false,
            clearance: HoseClearance::RadialAxis {
                link: 3,
                axis: Vec3::Y,
                min_radius: 0.090,
                bias_local: Vec3::Z,
            },
        },
        // 7-8: Hero elbow service loop: Link 3 top clamp -> Link 4 forearm spine across elbow_pitch
        DynamicHoseSpec {
            link_a: 3,
            pos_a: Vec3::new(0.015, 0.265, 0.091),
            tan_a: Vec3::new(0.0, 0.80, 0.60).normalize(),
            link_b: 4,
            pos_b: Vec3::new(0.012, 0.080, -0.11),
            tan_b: Vec3::NEG_Z,
            radius: 0.0125,
            slack: 0.62,
            uv_v_scale: 0.75,
            is_conduit: true,
            clearance: HoseClearance::ElbowCounterweight { link4: 4 },
        },
        DynamicHoseSpec {
            link_a: 3,
            pos_a: Vec3::new(-0.012, 0.265, 0.091),
            tan_a: Vec3::new(0.0, 0.80, 0.60).normalize(),
            link_b: 4,
            pos_b: Vec3::new(-0.011, 0.078, -0.11),
            tan_b: Vec3::NEG_Z,
            radius: 0.0056,
            slack: 0.62,
            uv_v_scale: 0.75,
            is_conduit: false,
            clearance: HoseClearance::ElbowCounterweight { link4: 4 },
        },
        // 9-10: Wrist-1 roll conduit & air hose: Link 4 front clamp -> Link 5 saddle/Y-splitter
        DynamicHoseSpec {
            link_a: 4,
            pos_a: Vec3::new(0.012, 0.080, -0.49),
            tan_a: Vec3::NEG_Z,
            link_b: 5,
            pos_b: Vec3::new(0.009, 0.052, -0.022),
            tan_b: Vec3::NEG_Z,
            radius: 0.0105,
            slack: 0.30,
            uv_v_scale: 0.35,
            is_conduit: true,
            clearance: HoseClearance::RadialAxis {
                link: 5,
                axis: Vec3::Z,
                min_radius: 0.050,
                bias_local: Vec3::Y,
            },
        },
        DynamicHoseSpec {
            link_a: 4,
            pos_a: Vec3::new(-0.011, 0.078, -0.49),
            tan_a: Vec3::NEG_Z,
            link_b: 5,
            pos_b: Vec3::new(-0.009, 0.052, -0.022),
            tan_b: Vec3::NEG_Z,
            radius: 0.0052,
            slack: 0.30,
            uv_v_scale: 0.35,
            is_conduit: false,
            clearance: HoseClearance::RadialAxis {
                link: 5,
                axis: Vec3::Z,
                min_radius: 0.050,
                bias_local: Vec3::Y,
            },
        },
        // 11-12: Twin pneumatic gripper jumpers (Link 5 Y-splitter -> Link 8 brass fittings across wrist_2, wrist_yaw, flange_roll)
        DynamicHoseSpec {
            link_a: 5,
            pos_a: Vec3::new(-0.011, 0.052, -0.032),
            tan_a: Vec3::new(-0.22, 0.25, -0.94).normalize(),
            link_b: 8,
            pos_b: Vec3::new(-0.044, 0.028, -0.010),
            tan_b: Vec3::NEG_Y,
            radius: 0.0036,
            slack: 0.36,
            uv_v_scale: 0.25,
            is_conduit: false,
            clearance: HoseClearance::RadialAxis {
                link: 6,
                axis: Vec3::X,
                min_radius: 0.042,
                bias_local: Vec3::Y,
            },
        },
        DynamicHoseSpec {
            link_a: 5,
            pos_a: Vec3::new(0.011, 0.052, -0.032),
            tan_a: Vec3::new(0.22, 0.25, -0.94).normalize(),
            link_b: 8,
            pos_b: Vec3::new(0.044, 0.028, -0.010),
            tan_b: Vec3::NEG_Y,
            radius: 0.0036,
            slack: 0.36,
            uv_v_scale: 0.25,
            is_conduit: false,
            clearance: HoseClearance::RadialAxis {
                link: 6,
                axis: Vec3::X,
                min_radius: 0.042,
                bias_local: Vec3::Y,
            },
        },
    ]
}

fn spawn_dynamic_joint_hoses(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    mats: &RobotMaterials,
    link_xforms: &[LinkTransform],
) {
    let rings = 16usize;
    let sides = 10usize;
    for spec in build_joint_hose_specs() {
        let mut mesh = allocate_tube_mesh(rings, sides);
        let hose = DynamicJointHose {
            link_a: spec.link_a,
            pos_a: spec.pos_a,
            tan_a: spec.tan_a,
            link_b: spec.link_b,
            pos_b: spec.pos_b,
            tan_b: spec.tan_b,
            radius: spec.radius,
            rings,
            sides,
            slack: spec.slack,
            uv_v_scale: spec.uv_v_scale,
            clearance: spec.clearance,
            mesh: Handle::default(),
        };
        let pts = evaluate_hose_centerline(&hose, link_xforms);
        write_tube_mesh_in_place(&mut mesh, &pts, hose.radius, hose.sides, hose.uv_v_scale);
        let mesh_handle = meshes.add(mesh);
        let mat = if spec.is_conduit {
            mats.conduit_straight.clone()
        } else {
            mats.air_hose.clone()
        };
        commands.spawn((
            DynamicJointHose {
                mesh: mesh_handle.clone(),
                ..hose
            },
            Mesh3d(mesh_handle),
            MeshMaterial3d(mat),
            Transform::IDENTITY,
            NoFrustumCulling,
        ));
    }
}

fn spawn_dynamic_drag_chain(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    mats: &RobotMaterials,
    rail_z: f32,
    initial_q0: f32,
) {
    let samples = 24usize;
    let mut mesh = allocate_drag_chain_mesh(samples);
    let chain = DynamicDragChain {
        rail_z,
        tray_x_anchor: 1.68,
        chain_len: 3.85,
        bend_radius: 0.045,
        samples,
        mesh: Handle::default(),
    };
    let pts = evaluate_drag_chain_centerline(&chain, initial_q0);
    write_drag_chain_mesh_in_place(&mut mesh, &pts);
    let mesh_handle = meshes.add(mesh);
    commands.spawn((
        DynamicDragChain {
            mesh: mesh_handle.clone(),
            ..chain
        },
        Mesh3d(mesh_handle),
        MeshMaterial3d(mats.conduit_straight.clone()),
        Transform::IDENTITY,
        NoFrustumCulling,
    ));
}

pub fn evaluate_hose_centerline(hose: &DynamicJointHose, link_xforms: &[LinkTransform]) -> Vec<Vec3> {
    let xa = link_xforms[hose.link_a];
    let xb = link_xforms[hose.link_b];
    let p0 = xa.pos + xa.rot * hose.pos_a;
    let p1 = xb.pos + xb.rot * hose.pos_b;
    let d0 = (xa.rot * hose.tan_a).normalize_or(Vec3::Y);
    let d1 = (xb.rot * hose.tan_b).normalize_or(Vec3::Y);

    let chord = p0.distance(p1).max(0.015);
    let m_len = chord * (0.48 + hose.slack);
    let m0 = d0 * m_len;
    let m1 = d1 * m_len;

    let rings = hose.rings.max(2);
    let mut pts = Vec::with_capacity(rings);
    for i in 0..rings {
        let t = i as f32 / (rings - 1) as f32;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        let mut pt = p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;

        if i > 0 && i + 1 < rings {
            let blend = (t * 5.0).min((1.0 - t) * 5.0).clamp(0.0, 1.0);
            match hose.clearance {
                HoseClearance::None => {}
                HoseClearance::RadialAxis {
                    link,
                    axis,
                    min_radius,
                    bias_local,
                } => {
                    let xf = link_xforms[link];
                    let inv_rot = xf.rot.inverse();
                    let local = inv_rot * (pt - xf.pos);
                    let ax = axis.normalize_or(Vec3::Y);
                    let proj = ax * local.dot(ax);
                    let rad_vec = local - proj;
                    let r_cur = rad_vec.length();
                    let req_r = min_radius + hose.radius;
                    let bias_perp = (bias_local - ax * bias_local.dot(ax)).normalize_or(Vec3::Z);
                    let dir = if r_cur > 1e-4 {
                        let d = rad_vec / r_cur;
                        if d.dot(bias_perp) < -0.15 {
                            bias_perp
                        } else {
                            d
                        }
                    } else {
                        bias_perp
                    };
                    if r_cur < req_r || rad_vec.dot(bias_perp) < 0.0 {
                        let pushed = proj + dir * r_cur.max(req_r);
                        let resolved = local.lerp(pushed, blend);
                        pt = xf.pos + xf.rot * resolved;
                    }
                }
                HoseClearance::ElbowCounterweight { link4 } => {
                    let xf = link_xforms[link4];
                    let inv_rot = xf.rot.inverse();
                    let mut local = inv_rot * (pt - xf.pos);
                    let r_pad = hose.radius;

                    // 1. Keep outside elbow drum (cylinder along X, r = 0.082)
                    let drum_r = 0.082 + r_pad;
                    let yz = Vec2::new(local.y, local.z);
                    let yz_len = yz.length();
                    if yz_len < drum_r {
                        let dir_yz = if yz_len > 1e-4 {
                            let d = yz / yz_len;
                            if d.x < -0.25 && d.y < 0.0 {
                                Vec2::new(0.70, 0.71)
                            } else {
                                d
                            }
                        } else {
                            Vec2::new(0.70, 0.71)
                        };
                        let pushed_yz = dir_yz * drum_r;
                        local.y = local.y + (pushed_yz.x - local.y) * blend;
                        local.z = local.z + (pushed_yz.y - local.z) * blend;
                    }

                    // 2. Keep outside rear counterweight housing (x ±0.068, y -0.056..0.088, z -0.015..0.168)
                    let bx = 0.068 + r_pad;
                    let by_min = -0.056 - r_pad;
                    let by_max = 0.088 + r_pad;
                    let bz_min = -0.015;
                    let bz_max = 0.168 + r_pad;
                    if local.x.abs() < bx
                        && local.y > by_min
                        && local.y < by_max
                        && local.z > bz_min
                        && local.z < bz_max
                    {
                        let dy = by_max - local.y;
                        let dz = bz_max - local.z;
                        if dy <= dz {
                            local.y += (by_max - local.y) * blend;
                        } else {
                            local.z += (bz_max - local.z) * blend;
                        }
                    }

                    pt = xf.pos + xf.rot * local;
                }
            }
        }
        pts.push(pt);
    }
    pts
}

pub fn evaluate_drag_chain_centerline(chain: &DynamicDragChain, q0: f32) -> Vec<(Vec3, Vec3)> {
    // Tray lies at Z = rail_z - 0.172, bottom support at y = 0.065, carriage entry at y = 0.155
    let z = chain.rail_z - 0.172;
    let r = chain.bend_radius;
    let y_bot = 0.065_f32;
    let y_top = y_bot + 2.0 * r;
    let x_anchor = chain.tray_x_anchor;
    let x_bend = (0.5 * (x_anchor + q0 + PI * r - chain.chain_len))
        .clamp(-1.95, q0.min(x_anchor) - 0.10);

    let count = chain.samples.max(8);
    let n_arc = (count / 2).max(6).min(count - 2);
    let n_bot = (count - n_arc) / 2;
    let n_top = count - n_arc - n_bot;

    let mut out = Vec::with_capacity(count);
    for i in 0..n_bot {
        let frac = i as f32 / (n_bot as f32);
        let x = x_anchor - frac * (x_anchor - x_bend);
        out.push((Vec3::new(x, y_bot, z), Vec3::NEG_X));
    }
    for k in 0..n_arc {
        let theta = PI * (k as f32 / (n_arc - 1) as f32);
        let cx = x_bend - r * theta.sin();
        let cy = y_bot + r * (1.0 - theta.cos());
        let tan = Vec3::new(-theta.cos(), theta.sin(), 0.0).normalize();
        out.push((Vec3::new(cx, cy, z), tan));
    }
    for k in 1..=n_top {
        let frac = k as f32 / (n_top as f32);
        let x = x_bend + frac * (q0 - x_bend);
        out.push((Vec3::new(x, y_top, z), Vec3::X));
    }
    out
}

fn allocate_tube_mesh(rings: usize, sides: usize) -> Mesh {
    let vert_count = rings * (sides + 1);
    let mut indices = Vec::with_capacity((rings - 1) * sides * 6);
    let stride = (sides + 1) as u32;
    for i in 0..(rings - 1) as u32 {
        let r0 = i * stride;
        let r1 = (i + 1) * stride;
        for j in 0..sides as u32 {
            let a = r0 + j;
            let b = r1 + j;
            let c = r0 + j + 1;
            let d = r1 + j + 1;
            indices.extend_from_slice(&[a, c, b, c, d, b]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32; 3]; vert_count])
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0_f32, 1.0, 0.0]; vert_count])
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0_f32; 2]; vert_count])
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1.0_f32, 0.0, 0.0, 1.0]; vert_count])
    .with_inserted_indices(Indices::U32(indices))
}

fn write_tube_mesh_in_place(
    mesh: &mut Mesh,
    points: &[Vec3],
    radius: f32,
    sides: usize,
    uv_v_scale: f32,
) {
    let rings = points.len();
    if rings < 2 {
        return;
    }

    let mut tangents = Vec::with_capacity(rings);
    for i in 0..rings {
        let t = if i == 0 {
            (points[1] - points[0]).normalize_or(Vec3::Y)
        } else if i + 1 == rings {
            (points[i] - points[i - 1]).normalize_or(Vec3::Y)
        } else {
            (points[i + 1] - points[i - 1]).normalize_or(Vec3::Y)
        };
        tangents.push(t);
    }

    // Initialize parallel-transport Bishop frame
    let t0 = tangents[0];
    let ref_hint = if t0.dot(Vec3::X).abs() < 0.85 {
        Vec3::X
    } else {
        Vec3::Z
    };
    let mut u_ax = (ref_hint - t0 * ref_hint.dot(t0)).normalize_or(Vec3::Z);

    let mut total_s = 0.0_f32;
    let stride = sides + 1;
    let expected = rings * stride;

    // Collect into flat attribute vectors or write in place
    let mut new_pos = Vec::with_capacity(expected);
    let mut new_norm = Vec::with_capacity(expected);
    let mut new_uv = Vec::with_capacity(expected);
    let mut new_tan = Vec::with_capacity(expected);

    for i in 0..rings {
        let ti = tangents[i];
        if i > 0 {
            total_s += points[i].distance(points[i - 1]);
            let t_prev = tangents[i - 1];
            if t_prev.dot(ti) > -0.9999 {
                u_ax = Quat::from_rotation_arc(t_prev, ti) * u_ax;
            }
            u_ax = (u_ax - ti * u_ax.dot(ti)).normalize_or(Vec3::X);
        }
        let v_ax = ti.cross(u_ax).normalize_or(Vec3::Z);
        let v_uv = total_s * (uv_v_scale / 0.15).max(0.5);

        for j in 0..=sides {
            let u_uv = j as f32 / sides as f32;
            let theta = TAU * u_uv;
            let (st, ct) = theta.sin_cos();
            let n = (u_ax * ct + v_ax * st).normalize();
            let d_theta = (-u_ax * st + v_ax * ct).normalize();
            let p = points[i] + n * radius;

            new_pos.push([p.x, p.y, p.z]);
            new_norm.push([n.x, n.y, n.z]);
            new_uv.push([u_uv, v_uv]);
            new_tan.push([d_theta.x, d_theta.y, d_theta.z, -1.0]);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, new_norm);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, new_uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, new_tan);
}

fn allocate_drag_chain_mesh(samples: usize) -> Mesh {
    allocate_tube_mesh(samples, 8)
}

fn write_drag_chain_mesh_in_place(mesh: &mut Mesh, centerline: &[(Vec3, Vec3)]) {
    let rings = centerline.len();
    if rings < 2 {
        return;
    }
    const SIDES: usize = 8;
    const HW: f32 = 0.019;
    const HH: f32 = 0.009;
    // Chamfered rectangular Igus-style link profile in (z_local, n_local)
    const PROFILE: [[f32; 2]; 8] = [
        [HW, HH * 0.65],
        [HW * 0.72, HH],
        [-HW * 0.72, HH],
        [-HW, HH * 0.65],
        [-HW, -HH * 0.65],
        [-HW * 0.72, -HH],
        [HW * 0.72, -HH],
        [HW, -HH * 0.65],
    ];

    let vert_count = rings * (SIDES + 1);
    let mut new_pos = Vec::with_capacity(vert_count);
    let mut new_norm = Vec::with_capacity(vert_count);
    let mut new_uv = Vec::with_capacity(vert_count);
    let mut new_tan = Vec::with_capacity(vert_count);

    let mut total_s = 0.0_f32;
    for (i, &(pos, tan)) in centerline.iter().enumerate() {
        if i > 0 {
            total_s += pos.distance(centerline[i - 1].0);
        }
        let u_ax = Vec3::Z;
        let v_ax = tan.cross(u_ax).normalize_or(Vec3::Y);
        let v_uv = total_s * 6.0;

        for j in 0..=SIDES {
            let idx = j % SIDES;
            let [pz, pn] = PROFILE[idx];
            let offset = u_ax * pz + v_ax * pn;
            let p = pos + offset;
            let n = offset.normalize_or(Vec3::Y);
            let d_theta = tan.cross(n).normalize_or(Vec3::Z);

            new_pos.push([p.x, p.y, p.z]);
            new_norm.push([n.x, n.y, n.z]);
            new_uv.push([j as f32 / SIDES as f32, v_uv]);
            new_tan.push([d_theta.x, d_theta.y, d_theta.z, -1.0]);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, new_norm);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, new_uv);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, new_tan);
}

fn update_dynamic_dress_pack(
    spray_job: Res<SprayJobRes>,
    mut meshes: ResMut<Assets<Mesh>>,
    hose_q: Query<&DynamicJointHose>,
    drag_q: Query<&DynamicDragChain>,
) {
    let joints = &spray_job.current_frame.joints;
    let link_xforms = spray_job.chain.fk_links(joints);
    let q0 = joints[0] as f32;

    for hose in &hose_q {
        if let Some(mut mesh) = meshes.get_mut(&hose.mesh) {
            let pts = evaluate_hose_centerline(hose, &link_xforms);
            write_tube_mesh_in_place(&mut mesh, &pts, hose.radius, hose.sides, hose.uv_v_scale);
        }
    }

    for chain in &drag_q {
        if let Some(mut mesh) = meshes.get_mut(&chain.mesh) {
            let pts = evaluate_drag_chain_centerline(chain, q0);
            write_drag_chain_mesh_in_place(&mut mesh, &pts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gripper_jaws_clear_can_radius_and_close_flush() {
        let closed_l = jaw_local_translation(-1.0, 0.0);
        let closed_r = jaw_local_translation(1.0, 0.0);
        // Jaw thickness is 0.012m -> inner face is at |x| - 0.006m = 0.033m (exact can radius)
        assert!((closed_r.x - 0.006 - 0.033).abs() < 1e-5);
        assert!((closed_l.x + 0.006 + 0.033).abs() < 1e-5);

        let open_r = jaw_local_translation(1.0, 1.0);
        assert!(open_r.x - 0.006 > 0.045);
    }

    #[test]
    fn trigger_finger_depresses_downward_toward_nozzle() {
        let released = trigger_local_transform(0.0);
        let pressed = trigger_local_transform(1.0);
        let tip_local = Vec3::new(0.0, -0.012, -0.022);
        let y_rel = (released * tip_local).y;
        let y_dep = (pressed * tip_local).y;
        assert!(y_dep < y_rel - 0.004);
    }

    #[test]
    fn procedural_robot_textures_and_tangent_meshes_are_valid() {
        for (a, rm, n) in [
            build_cast_iron_textures(64),
            build_safety_orange_textures(64, 0.0),
            build_safety_orange_textures(64, 1.0),
            build_anodized_dark_textures(64),
            build_brushed_aluminum_textures(64),
            build_graphite_textures(64),
            build_rubber_grip_textures(64),
            build_conduit_textures(64, true),
        ] {
            for img in [&a, &rm, &n] {
                assert_eq!(img.texture_descriptor.size.width, 64);
                assert_eq!(img.texture_descriptor.size.height, 64);
                assert_eq!(img.data.as_ref().map(|d| d.len()), Some(64 * 64 * 4));
            }
        }

        let unit07 = build_unit07_stencil_texture(512, 192);
        assert_eq!(unit07.texture_descriptor.size.width, 512);
        assert_eq!(unit07.texture_descriptor.size.height, 192);
        assert_eq!(unit07.data.as_ref().map(|d| d.len()), Some(512 * 192 * 4));

        let chev = build_hazard_chevron_texture(256, 64);
        assert_eq!(chev.texture_descriptor.size.width, 256);
        assert_eq!(chev.texture_descriptor.size.height, 64);

        let dial = build_encoder_dial_texture(128);
        assert_eq!(dial.texture_descriptor.size.width, 128);
        assert_eq!(dial.texture_descriptor.size.height, 128);

        let scale = build_degree_scale_collar_texture(256, 32);
        assert_eq!(scale.texture_descriptor.size.width, 256);
        assert_eq!(scale.texture_descriptor.size.height, 32);

        let (can_a, can_rm) = build_spray_can_label_textures([232, 93, 26], 0, 128);
        assert_eq!(can_a.texture_descriptor.size.width, 128);
        assert_eq!(can_rm.texture_descriptor.size.height, 128);
        assert_eq!(can_a.data.as_ref().map(|d| d.len()), Some(128 * 128 * 4));

        // Verify tangent generation succeeds on every primitive type used on the robot
        let meshes_with_tangents = [
            with_tangents(Cuboid::new(0.2, 0.1, 0.1).mesh().build()),
            with_tangents(Cylinder::new(0.05, 0.2).mesh().resolution(16).build()),
            with_tangents(
                ConicalFrustum {
                    radius_top: 0.04,
                    radius_bottom: 0.06,
                    height: 0.2,
                }
                .mesh()
                .resolution(16)
                .build(),
            ),
            with_tangents(
                Torus {
                    minor_radius: 0.01,
                    major_radius: 0.05,
                }
                .mesh()
                .minor_resolution(12)
                .major_resolution(16)
                .angle_range(0.0..=PI)
                .build(),
            ),
            make_decal_quad_mesh(0.12, 0.04),
        ];
        for mesh in meshes_with_tangents {
            assert!(
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_some(),
                "expected tangent attribute on mesh"
            );
        }
    }

    #[test]
    fn spawns_all_nine_robot_links_matching_fk_and_end_effector_on_link_8() {
        use clap::Parser;

        use crate::cli::CliArgs;
        use crate::core::types::StudioConfig;

        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.insert_resource(Assets::<Image>::default());
        app.insert_resource(AppConfig::new(CliArgs::parse_from(["spraybot"])));
        let studio = StudioConfig::default();
        let job_res = SprayJobRes::empty(&studio);
        let expected_links = job_res.chain.fk_links(&job_res.current_frame.joints);
        app.insert_resource(job_res);
        app.add_plugins(plugin);
        app.update();

        let world = app.world_mut();
        let mut link_q = world.query::<(Entity, &RobotLink, &Transform)>();
        let mut seen = vec![false; 9];
        let mut link8_entity = None;
        for (entity, link, tf) in link_q.iter(world) {
            assert!(link.index < 9, "unexpected RobotLink index {}", link.index);
            assert!(!seen[link.index], "duplicate RobotLink index {}", link.index);
            seen[link.index] = true;
            let expected = expected_links[link.index];
            assert!(
                (tf.translation - expected.pos).length() < 1e-5,
                "link {} pos mismatch: {:?} vs {:?}",
                link.index,
                tf.translation,
                expected.pos
            );
            assert!(
                tf.rotation.dot(expected.rot).abs() > 1.0 - 1e-5,
                "link {} rot mismatch: {:?} vs {:?} (angle_between={:?})",
                link.index,
                tf.rotation,
                expected.rot,
                tf.rotation.angle_between(expected.rot)
            );
            if link.index == 8 {
                link8_entity = Some(entity);
            }
        }
        assert!(seen.iter().all(|&v| v), "missing RobotLink in 0..9: {seen:?}");

        let link8 = link8_entity.expect("link 8 must exist");
        let mut jaw_q = world.query::<(&GripperJaw, &ChildOf)>();
        let jaw_parents: Vec<Entity> = jaw_q.iter(world).map(|(_, p)| p.parent()).collect();
        assert_eq!(jaw_parents, vec![link8, link8]);

        let mut trigger_q = world.query_filtered::<&ChildOf, With<TriggerFinger>>();
        let trigger_parents: Vec<Entity> = trigger_q.iter(world).map(|p| p.parent()).collect();
        assert_eq!(trigger_parents, vec![link8]);
    }

    #[test]
    fn dynamic_joint_hoses_stay_connected_and_clear_joints_across_extreme_poses() {
        use crate::core::kinematics::KinematicChain;
        use crate::core::types::StudioConfig;

        let studio = StudioConfig::default();
        let chain = KinematicChain::default_spraybot(&studio);
        let poses: [[f64; 9]; 5] = [
            chain.comfort_pose,
            [1.38, -0.25, 0.15, 0.0, 1.10, 0.0, -1.25, 0.0, 0.0],
            [-1.40, 1.35, -1.15, 1.55, -1.85, 1.40, -1.30, 1.10, 2.20],
            [0.85, -1.45, 0.85, -1.55, 1.95, -1.60, 1.15, -1.15, -2.10],
            [-1.70, 2.60, -0.35, 0.65, 1.45, -0.75, 0.45, 0.85, 0.90],
        ];

        for q in &poses {
            let xforms = chain.fk_links(q);
            for (idx, spec) in build_joint_hose_specs().into_iter().enumerate() {
                let hose = DynamicJointHose {
                    link_a: spec.link_a,
                    pos_a: spec.pos_a,
                    tan_a: spec.tan_a,
                    link_b: spec.link_b,
                    pos_b: spec.pos_b,
                    tan_b: spec.tan_b,
                    radius: spec.radius,
                    rings: 16,
                    sides: 10,
                    slack: spec.slack,
                    uv_v_scale: spec.uv_v_scale,
                    clearance: spec.clearance,
                    mesh: Handle::default(),
                };
                let pts = evaluate_hose_centerline(&hose, &xforms);
                assert_eq!(pts.len(), 16);

                let expected_a = xforms[spec.link_a].pos + xforms[spec.link_a].rot * spec.pos_a;
                let expected_b = xforms[spec.link_b].pos + xforms[spec.link_b].rot * spec.pos_b;
                assert!(
                    (pts[0] - expected_a).length() < 1e-4,
                    "hose {idx} detached from start anchor on link {} at pose {q:?}",
                    spec.link_a
                );
                assert!(
                    (pts[15] - expected_b).length() < 1e-4,
                    "hose {idx} detached from end anchor on link {} at pose {q:?}",
                    spec.link_b
                );
                for (k, &p) in pts.iter().enumerate() {
                    assert!(p.is_finite(), "hose {idx} pt {k} non-finite");
                    assert!(
                        p.z > 0.02,
                        "hose {idx} pt {k} z={:.4} came too close to canvas",
                        p.z
                    );
                }

                // Interior points of ElbowCounterweight hose must clear Link 4 drum & counterweight
                if let HoseClearance::ElbowCounterweight { link4 } = spec.clearance {
                    let xf4 = xforms[link4];
                    let inv4 = xf4.rot.inverse();
                    for &p in &pts[3..13] {
                        let loc = inv4 * (p - xf4.pos);
                        let yz_r = (loc.y * loc.y + loc.z * loc.z).sqrt();
                        assert!(
                            yz_r >= 0.080 + spec.radius - 1e-3,
                            "elbow hose clipped Link 4 drum: yz_r={yz_r:.4}"
                        );
                    }
                }
            }
        }

        // Drag chain length & bounds check across full rail range
        let dc = DynamicDragChain {
            rail_z: studio.rail_z,
            tray_x_anchor: 1.68,
            chain_len: 3.85,
            bend_radius: 0.045,
            samples: 24,
            mesh: Handle::default(),
        };
        for q0 in [studio.rail_q0_min, -0.5, 0.0, 0.8, studio.rail_q0_max] {
            let pts = evaluate_drag_chain_centerline(&dc, q0);
            assert_eq!(pts.len(), 24);
            assert!((pts[0].0.x - 1.68).abs() < 1e-4);
            assert!((pts[23].0.x - q0).abs() < 1e-4);
            for (p, _) in &pts {
                assert!(p.x >= -2.01 && p.x <= 1.72, "drag chain out of tray: {}", p.x);
                assert!(p.y >= 0.06 && p.y <= 0.16, "drag chain y out of bounds: {}", p.y);
            }
        }
    }
}
