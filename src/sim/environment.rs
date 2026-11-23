use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    asset::RenderAssetUsages,
    light::{
        CascadeShadowConfigBuilder, FogVolume, GeneratedEnvironmentMapLight, GlobalAmbientLight,
        NotShadowCaster, NotShadowReceiver, Skybox, VolumetricLight,
    },
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
};

use crate::core::types::StudioConfig;
use crate::sim::state::AppConfig;
use crate::sim::textures::{
    MipmapKind, create_mipmapped_pbr_triplet, create_mipmapped_rgba8_image,
};

pub fn plugin(app: &mut App) {
    app.add_systems(Startup, setup_studio_environment)
        .add_systems(PreUpdate, attach_studio_env_map_to_cameras);
}

#[derive(Resource)]
struct StudioEnvMap {
    cubemap: Handle<Image>,
}

fn attach_studio_env_map_to_cameras(
    mut commands: Commands,
    env_map: Option<Res<StudioEnvMap>>,
    cameras: Query<Entity, (With<Camera3d>, Without<GeneratedEnvironmentMapLight>)>,
) {
    let Some(env_map) = env_map else {
        return;
    };
    for cam in &cameras {
        commands.entity(cam).insert((
            GeneratedEnvironmentMapLight {
                environment_map: env_map.cubemap.clone(),
                intensity: 450.0,
                ..default()
            },
            Skybox {
                image: Some(env_map.cubemap.clone()),
                brightness: 450.0,
                ..default()
            },
        ));
    }
}

fn setup_studio_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    app_config: Res<AppConfig>,
) {
    let studio = &app_config.studio;

    // 1. Procedural HDR studio reflection cubemap (128x128 per face for crisp metallic tube highlights)
    let cubemap_handle = images.add(build_studio_hdr_cubemap(128));
    commands.insert_resource(StudioEnvMap {
        cubemap: cubemap_handle,
    });

    // 2. Studio room shell: oversprayed concrete floor, drop-cloth runner, ribbed walls/ceiling, perimeter props
    spawn_studio_room(&mut commands, &mut meshes, &mut materials, &mut images);

    // 3. Wooden canvas stretcher frame + backing behind Z = 0.00m
    spawn_canvas_stretcher(&mut commands, &mut meshes, &mut materials, &mut images, studio);

    // 4. Machined 3.80m steel linear floor rail at Z = 0.95m
    spawn_floor_rail(&mut commands, &mut meshes, &mut materials, studio);

    // 5. 2-tier 20°-yawed steel spray can rack matching StudioConfig::rack_slots to the millimeter
    spawn_spray_can_rack(
        &mut commands,
        &mut meshes,
        &mut materials,
        studio,
        app_config.cli.max_cans,
    );

    // 6. Studio lighting (cool skylight, neutral canvas key, cool rack key, warm amber side work-light, ceiling tube fixtures, volumetric haze)
    spawn_studio_lights(&mut commands, &mut meshes, &mut materials, &mut images);
}

fn spawn_studio_room(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
) {
    let (floor_albedo, floor_roughness, floor_normal) = build_concrete_floor_textures(1024);
    let floor_albedo_h = images.add(floor_albedo);
    let floor_roughness_h = images.add(floor_roughness);
    let floor_normal_h = images.add(floor_normal);

    let floor_mesh = meshes.add(
        Plane3d::default()
            .mesh()
            .size(8.0, 6.5)
            .build()
            .with_generated_tangents()
            .expect("floor tangents"),
    );
    let floor_mat = materials.add(StandardMaterial {
        base_color_texture: Some(floor_albedo_h),
        metallic_roughness_texture: Some(floor_roughness_h),
        normal_map_texture: Some(floor_normal_h),
        metallic: 0.0,
        perceptual_roughness: 1.0,
        reflectance: 0.42,
        clearcoat: 0.28,
        clearcoat_perceptual_roughness: 0.26,
        ..default()
    });
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_mat),
        Transform::from_xyz(0.0, 0.0, 3.21),
    ));

    // Taped kraft-paper / canvas drop-cloth runner under the canvas (Z in [0.04, 0.72], clear of rail at Z = 0.95)
    spawn_drop_cloth_runner(commands, meshes, materials, images);

    // Back wall (Z = -0.04m front surface) with non-repeating weathered CMU block texture
    let (wall_albedo, wall_rough, wall_normal) = build_industrial_wall_textures(1280, 640);
    let wall_albedo_h = images.add(wall_albedo);
    let wall_rough_h = images.add(wall_rough);
    let wall_normal_h = images.add(wall_normal);

    let back_wall_mesh = meshes.add(
        Cuboid::new(8.0, 4.2, 0.16)
            .mesh()
            .build()
            .with_generated_tangents()
            .expect("wall tangents"),
    );
    let back_wall_mat = materials.add(StandardMaterial {
        base_color_texture: Some(wall_albedo_h),
        metallic_roughness_texture: Some(wall_rough_h),
        normal_map_texture: Some(wall_normal_h),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        reflectance: 0.35,
        ..default()
    });
    commands.spawn((
        Mesh3d(back_wall_mesh),
        MeshMaterial3d(back_wall_mat),
        // Front face sits at -0.122 + 0.08 = -0.042m (cleanly flush behind stretcher back -0.041m)
        Transform::from_xyz(0.0, 2.1, -0.122),
    ));

    // Side walls (ribs along V = Z so corrugation runs vertically) & rear wall (ribs along U = X)
    let (side_albedo, side_rough, side_normal) = build_corrugated_panel_textures(512, true);
    let side_wall_mat = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(side_albedo)),
        metallic_roughness_texture: Some(images.add(side_rough)),
        normal_map_texture: Some(images.add(side_normal)),
        perceptual_roughness: 1.0,
        metallic: 0.18,
        ..default()
    });
    let side_wall_mesh = meshes.add(
        Cuboid::new(0.16, 4.2, 6.5)
            .mesh()
            .build()
            .with_generated_tangents()
            .expect("side wall tangents"),
    );
    commands.spawn((
        Mesh3d(side_wall_mesh.clone()),
        MeshMaterial3d(side_wall_mat.clone()),
        Transform::from_xyz(-4.08, 2.1, 3.21),
    ));
    commands.spawn((
        Mesh3d(side_wall_mesh),
        MeshMaterial3d(side_wall_mat),
        Transform::from_xyz(4.08, 2.1, 3.21),
    ));

    let (rear_albedo, rear_rough, rear_normal) = build_corrugated_panel_textures(512, false);
    let rear_wall_mat = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(rear_albedo)),
        metallic_roughness_texture: Some(images.add(rear_rough)),
        normal_map_texture: Some(images.add(rear_normal)),
        perceptual_roughness: 1.0,
        metallic: 0.18,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(
            Cuboid::new(8.0, 4.2, 0.16)
                .mesh()
                .build()
                .with_generated_tangents()
                .expect("rear wall tangents"),
        )),
        MeshMaterial3d(rear_wall_mat),
        Transform::from_xyz(0.0, 2.1, 6.54),
    ));

    // Dark industrial ribbed steel ceiling deck
    let (ceil_albedo, ceil_rough, ceil_normal) = build_ceiling_deck_textures(512);
    let ceiling_mat = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(ceil_albedo)),
        metallic_roughness_texture: Some(images.add(ceil_rough)),
        normal_map_texture: Some(images.add(ceil_normal)),
        perceptual_roughness: 1.0,
        metallic: 0.30,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(
            Cuboid::new(8.0, 0.16, 6.5)
                .mesh()
                .build()
                .with_generated_tangents()
                .expect("ceiling tangents"),
        )),
        MeshMaterial3d(ceiling_mat),
        Transform::from_xyz(0.0, 4.28, 3.21),
    ));

    // Skylight diffuser light panels recessed in ceiling
    let skylight_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.97, 1.0),
        emissive: LinearRgba::new(2.8, 3.0, 3.4, 1.0),
        unlit: true,
        ..default()
    });
    let panel_mesh = meshes.add(Cuboid::new(2.2, 0.03, 0.95));
    for &(px, pz) in &[(-1.6, 1.5), (1.6, 1.5), (-1.6, 3.8), (1.6, 3.8)] {
        commands.spawn((
            Mesh3d(panel_mesh.clone()),
            MeshMaterial3d(skylight_mat.clone()),
            Transform::from_xyz(px, 4.185, pz),
            NotShadowCaster,
        ));
    }

    // Structural gunmetal steel columns, baseboards & ceiling trusses
    let steel_struct_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.20, 0.23),
        metallic: 0.78,
        perceptual_roughness: 0.44,
        ..default()
    });

    // Baseboards (back & side walls)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 0.12, 0.03))),
        MeshMaterial3d(steel_struct_mat.clone()),
        Transform::from_xyz(0.0, 0.06, -0.027),
    ));
    let side_base_mesh = meshes.add(Cuboid::new(0.03, 0.12, 6.5));
    for &sx in &[-3.985, 3.985] {
        commands.spawn((
            Mesh3d(side_base_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(sx, 0.06, 3.21),
        ));
    }

    // Vertical structural steel columns along side walls and back wall corners
    let col_mesh = meshes.add(Cuboid::new(0.16, 4.2, 0.16));
    for &z in &[0.06, 2.15, 4.25, 6.25] {
        for &x in &[-3.92, 3.92] {
            commands.spawn((
                Mesh3d(col_mesh.clone()),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(x, 2.1, z),
            ));
        }
    }
    for &x in &[-3.65, 3.65] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.14, 4.2, 0.12))),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(x, 2.1, 0.02),
        ));
    }

    // Overhead steel trusses spanning X
    let chord_mesh = meshes.add(Cuboid::new(7.84, 0.08, 0.08));
    let web_mesh = meshes.add(Cuboid::new(0.05, 0.36, 0.05));
    for &tz in &[0.55, 2.20, 3.85, 5.50] {
        commands.spawn((
            Mesh3d(chord_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(0.0, 4.12, tz),
        ));
        commands.spawn((
            Mesh3d(chord_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(0.0, 3.74, tz),
        ));
        for ix in -3..=3 {
            commands.spawn((
                Mesh3d(web_mesh.clone()),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(ix as f32 * 1.1, 3.93, tz),
            ));
        }
    }

    // Hanging LED / fluorescent shop light fixtures suspended from overhead trusses (y ≈ 3.94m)
    spawn_overhead_shop_lights(commands, meshes, materials, &steel_struct_mat);

    // Subtle safety zone floor stripe framing the robot linear track envelope
    let hazard_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.62, 0.12),
        perceptual_roughness: 0.65,
        metallic: 0.05,
        ..default()
    });
    let stripe_x_mesh = meshes.add(Cuboid::new(4.30, 0.002, 0.035));
    let stripe_z_mesh = meshes.add(Cuboid::new(0.035, 0.002, 1.36));
    commands.spawn((
        Mesh3d(stripe_x_mesh.clone()),
        MeshMaterial3d(hazard_mat.clone()),
        Transform::from_xyz(-0.15, 0.001, 1.56),
    ));
    commands.spawn((
        Mesh3d(stripe_x_mesh),
        MeshMaterial3d(hazard_mat.clone()),
        Transform::from_xyz(-0.15, 0.001, 0.22),
    ));
    commands.spawn((
        Mesh3d(stripe_z_mesh.clone()),
        MeshMaterial3d(hazard_mat.clone()),
        Transform::from_xyz(-2.28, 0.001, 0.89),
    ));
    commands.spawn((
        Mesh3d(stripe_z_mesh),
        MeshMaterial3d(hazard_mat),
        Transform::from_xyz(1.98, 0.001, 0.89),
    ));

    // Perimeter workshop details strictly outside robot sweep envelope and camera sightlines
    spawn_perimeter_workshop_props(commands, meshes, materials, &steel_struct_mat);
}

fn spawn_canvas_stretcher(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
    studio: &crate::core::types::StudioConfig,
) {
    let (wood_a, wood_r, wood_n) = build_wood_grain_textures(256);
    let wood_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(wood_a)),
        metallic_roughness_texture: Some(images.add(wood_r)),
        normal_map_texture: Some(images.add(wood_n)),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        reflectance: 0.35,
        ..default()
    });
    let backing_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.56, 0.42, 0.27),
        perceptual_roughness: 0.78,
        metallic: 0.0,
        ..default()
    });
    let bracket_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.17, 0.19),
        metallic: 0.82,
        perceptual_roughness: 0.38,
        ..default()
    });
    let staple_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.79, 0.83),
        metallic: 0.90,
        perceptual_roughness: 0.24,
        ..default()
    });

    let w = studio.canvas_width; // 2.50m
    let h = studio.canvas_height; // 1.50m
    let d = studio.canvas_depth; // 0.04m
    let cx = studio.canvas_center.x;
    let cy = studio.canvas_center.y; // 1.55m
    let cz = -0.021; // front face at -0.001m, flush behind canvas.rs (Z = 0.001m)

    let frame_bar = 0.055;

    // Top and bottom stretcher rails
    let horiz_mesh = meshes.add(
        Cuboid::new(w, frame_bar, d)
            .mesh()
            .build()
            .with_generated_tangents()
            .expect("horiz stretcher tangents"),
    );
    commands.spawn((
        Mesh3d(horiz_mesh.clone()),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx, cy + h * 0.5 - frame_bar * 0.5, cz),
    ));
    commands.spawn((
        Mesh3d(horiz_mesh),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx, cy - h * 0.5 + frame_bar * 0.5, cz),
    ));

    // Left and right stretcher stiles
    let vert_mesh = meshes.add(
        Cuboid::new(frame_bar, h - 2.0 * frame_bar, d)
            .mesh()
            .build()
            .with_generated_tangents()
            .expect("vert stretcher tangents"),
    );
    commands.spawn((
        Mesh3d(vert_mesh.clone()),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx - w * 0.5 + frame_bar * 0.5, cy, cz),
    ));
    commands.spawn((
        Mesh3d(vert_mesh),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx + w * 0.5 - frame_bar * 0.5, cy, cz),
    ));

    // Internal cross-braces
    commands.spawn((
        Mesh3d(
            meshes.add(
                Cuboid::new(0.045, h - 2.0 * frame_bar, d * 0.8)
                    .mesh()
                    .build()
                    .with_generated_tangents()
                    .expect("brace tangents"),
            ),
        ),
        MeshMaterial3d(wood_mat.clone()),
        Transform::from_xyz(cx, cy, cz - 0.003),
    ));
    commands.spawn((
        Mesh3d(
            meshes.add(
                Cuboid::new(w - 2.0 * frame_bar, 0.045, d * 0.8)
                    .mesh()
                    .build()
                    .with_generated_tangents()
                    .expect("brace tangents"),
            ),
        ),
        MeshMaterial3d(wood_mat),
        Transform::from_xyz(cx, cy, cz - 0.003),
    ));

    // Recessed solid backing board (front face at Z = -0.002m)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(w - 0.006, h - 0.006, 0.014))),
        MeshMaterial3d(backing_mat),
        Transform::from_xyz(cx, cy, -0.009),
    ));

    // Dark-steel corner gusset L-brackets & canvas-wrap staples along outer bevels (strictly z <= -0.003m)
    let gusset_h = meshes.add(Cuboid::new(0.085, 0.014, d + 0.003));
    let gusset_v = meshes.add(Cuboid::new(0.014, 0.085, d + 0.003));
    for &(sx, sy) in &[(-1.0_f32, 1.0_f32), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
        let ex = cx + sx * (w * 0.5 - 0.040);
        let ey = cy + sy * (h * 0.5 - 0.006);
        commands.spawn((
            Mesh3d(gusset_h.clone()),
            MeshMaterial3d(bracket_mat.clone()),
            Transform::from_xyz(ex, ey, cz - 0.002),
        ));
        let ex_v = cx + sx * (w * 0.5 - 0.006);
        let ey_v = cy + sy * (h * 0.5 - 0.040);
        commands.spawn((
            Mesh3d(gusset_v.clone()),
            MeshMaterial3d(bracket_mat.clone()),
            Transform::from_xyz(ex_v, ey_v, cz - 0.002),
        ));
    }

    let staple_side = meshes.add(Cuboid::new(0.0025, 0.011, 0.003));
    for &sx in &[-1.0_f32, 1.0] {
        let bx = cx + sx * (w * 0.5 + 0.0008);
        for k in 0..9 {
            let sy = cy - h * 0.42 + (k as f32 / 8.0) * (h * 0.84);
            commands.spawn((
                Mesh3d(staple_side.clone()),
                MeshMaterial3d(staple_mat.clone()),
                Transform::from_xyz(bx, sy, cz + 0.004),
            ));
        }
    }

    // Wall-mounting steel cleats
    let cleat_mesh = meshes.add(Cuboid::new(0.10, 0.035, 0.03));
    for &(bx, by) in &[
        (cx - 0.90, cy + h * 0.5 + 0.014),
        (cx + 0.90, cy + h * 0.5 + 0.014),
        (cx - 0.90, cy - h * 0.5 - 0.014),
        (cx + 0.90, cy - h * 0.5 - 0.014),
    ] {
        commands.spawn((
            Mesh3d(cleat_mesh.clone()),
            MeshMaterial3d(bracket_mat.clone()),
            Transform::from_xyz(bx, by, cz - 0.005),
        ));
    }
}

fn spawn_floor_rail(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    studio: &crate::core::types::StudioConfig,
) {
    // 3.80m rail: X in [-2.05, +1.75], center X = -0.15m, Z = studio.rail_z (0.95m)
    let rail_len = studio.rail_length;
    let rail_center_x = -0.15;
    let rz = studio.rail_z;

    let bed_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.14, 0.15, 0.17),
        metallic: 0.72,
        perceptual_roughness: 0.48,
        ..default()
    });
    let machined_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.79, 0.82),
        metallic: 0.93,
        perceptual_roughness: 0.18,
        ..default()
    });
    let screw_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.54, 0.56, 0.60),
        metallic: 0.88,
        perceptual_roughness: 0.28,
        ..default()
    });
    let bumper_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#E85D1A").unwrap().into(),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        ..default()
    });

    // Main cast-iron rail bed
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(rail_len, 0.086, 0.28))),
        MeshMaterial3d(bed_mat.clone()),
        Transform::from_xyz(rail_center_x, 0.059, rz),
    ));

    // Floor leveling / anchor foot plates
    let foot_mesh = meshes.add(Cuboid::new(0.16, 0.016, 0.38));
    for i in 0..6 {
        let fx = -1.95 + i as f32 * 0.72;
        commands.spawn((
            Mesh3d(foot_mesh.clone()),
            MeshMaterial3d(bed_mat.clone()),
            Transform::from_xyz(fx, 0.008, rz),
        ));
    }

    // Twin precision-ground linear guideways
    let guide_mesh = meshes.add(Cuboid::new(rail_len - 0.04, 0.022, 0.024));
    for &dz in &[-0.095, 0.095] {
        commands.spawn((
            Mesh3d(guide_mesh.clone()),
            MeshMaterial3d(machined_steel_mat.clone()),
            Transform::from_xyz(rail_center_x, 0.113, rz + dz),
        ));
    }

    // Central drive ball-screw shaft
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.014, rail_len - 0.12))),
        MeshMaterial3d(screw_steel_mat),
        Transform::from_xyz(rail_center_x, 0.112, rz)
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    ));

    // Cable drag-chain tray along back edge of rail
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(rail_len - 0.08, 0.036, 0.055))),
        MeshMaterial3d(bed_mat.clone()),
        Transform::from_xyz(rail_center_x, 0.038, rz - 0.172),
    ));

    // End-stop blocks and orange urethane bumpers
    let stop_mesh = meshes.add(Cuboid::new(0.065, 0.072, 0.26));
    let bump_mesh = meshes.add(Cuboid::new(0.020, 0.042, 0.055));
    for &(sx, bx) in &[(-2.015, -1.974), (1.715, 1.674)] {
        commands.spawn((
            Mesh3d(stop_mesh.clone()),
            MeshMaterial3d(bed_mat.clone()),
            Transform::from_xyz(sx, 0.138, rz),
        ));
        commands.spawn((
            Mesh3d(bump_mesh.clone()),
            MeshMaterial3d(bumper_mat.clone()),
            Transform::from_xyz(bx, 0.145, rz),
        ));
    }
}

fn spawn_spray_can_rack(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    studio: &crate::core::types::StudioConfig,
    max_cans: usize,
) {
    let slots = studio.rack_slots(max_cans);
    let upper_count = max_cans.div_ceil(2);
    let lower_count = max_cans - upper_count;
    let tier_slots = upper_count.max(lower_count).max(1);

    let rack_width = tier_slots as f32 * studio.rack_slot_spacing + 0.070;
    let half_w = rack_width * 0.5;
    let can_bottom_offset = -0.085; // matches CanEntity bottom rim in grasp frame

    let rack_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.18, 0.21),
        metallic: 0.78,
        perceptual_roughness: 0.40,
        ..default()
    });
    let cradle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.35, 0.40),
        metallic: 0.84,
        perceptual_roughness: 0.32,
        ..default()
    });

    let rack_rot = Quat::from_rotation_y(studio.rack_yaw_deg.to_radians());
    let rack_root = commands
        .spawn((
            Transform::from_translation(Vec3::new(
                studio.rack_center_xz.x,
                0.0,
                studio.rack_center_xz.z,
            ))
            .with_rotation(rack_rot),
            Visibility::default(),
        ))
        .id();

    let shelf_mesh = meshes.add(Cuboid::new(rack_width, 0.010, 0.092));
    let back_lip_mesh = meshes.add(Cuboid::new(rack_width, 0.032, 0.008));
    let front_lip_mesh = meshes.add(Cuboid::new(rack_width, 0.015, 0.007));
    let divider_mesh = meshes.add(Cuboid::new(0.007, 0.022, 0.076));

    let tiers = [
        (
            upper_count,
            studio.rack_upper_grasp_y + can_bottom_offset,
            -studio.rack_tier_step_n,
        ),
        (
            lower_count,
            studio.rack_lower_grasp_y + can_bottom_offset,
            studio.rack_tier_step_n,
        ),
    ];

    commands.entity(rack_root).with_children(|parent| {
        for (count, shelf_top_y, tier_z) in tiers {
            if count == 0 {
                continue;
            }
            // Shelf plate: top face sits cleanly at shelf_top_y = grasp_y - 0.085
            parent.spawn((
                Mesh3d(shelf_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(0.0, shelf_top_y - 0.005, tier_z),
            ));
            // Rear backstop rail
            parent.spawn((
                Mesh3d(back_lip_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(0.0, shelf_top_y + 0.016, tier_z - 0.042),
            ));
            // Front retention lip (0.015m high — cleared cleanly by the 0.06m PickCan vertical lift)
            parent.spawn((
                Mesh3d(front_lip_mesh.clone()),
                MeshMaterial3d(cradle_mat.clone()),
                Transform::from_xyz(0.0, shelf_top_y + 0.0075, tier_z + 0.0425),
            ));

            // Cradle divider fins between & outside each slot
            for d in 0..=count {
                let x_div =
                    (d as f32 - count as f32 * 0.5) * studio.rack_slot_spacing;
                parent.spawn((
                    Mesh3d(divider_mesh.clone()),
                    MeshMaterial3d(cradle_mat.clone()),
                    Transform::from_xyz(x_div, shelf_top_y + 0.011, tier_z),
                ));
            }
        }

        // Left & right welded steel side frames + floor baseplates
        let leg_mesh = meshes.add(Cuboid::new(0.036, 1.20, 0.050));
        let front_leg_mesh = meshes.add(Cuboid::new(0.034, 0.86, 0.044));
        let foot_mesh = meshes.add(Cuboid::new(0.090, 0.014, 0.24));
        let gusset_mesh = meshes.add(Cuboid::new(0.034, 0.045, 0.16));

        for &side in &[-1.0_f32, 1.0] {
            let x_leg = side * (half_w - 0.012);
            // Rear upright supporting upper tier
            parent.spawn((
                Mesh3d(leg_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.60, -studio.rack_tier_step_n - 0.025),
            ));
            // Front upright supporting lower tier
            parent.spawn((
                Mesh3d(front_leg_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.43, studio.rack_tier_step_n + 0.015),
            ));
            // Floor mounting plate
            parent.spawn((
                Mesh3d(foot_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.007, 0.0),
            ));
            // Mid tie gusset
            parent.spawn((
                Mesh3d(gusset_mesh.clone()),
                MeshMaterial3d(rack_steel_mat.clone()),
                Transform::from_xyz(x_leg, 0.52, 0.0),
            ));
        }

        // Rear cross-braces connecting left & right frames
        let brace_mesh = meshes.add(Cuboid::new(rack_width - 0.02, 0.032, 0.032));
        parent.spawn((
            Mesh3d(brace_mesh.clone()),
            MeshMaterial3d(rack_steel_mat.clone()),
            Transform::from_xyz(0.0, 0.28, -studio.rack_tier_step_n - 0.025),
        ));
        parent.spawn((
            Mesh3d(brace_mesh),
            MeshMaterial3d(rack_steel_mat.clone()),
            Transform::from_xyz(0.0, 0.72, -studio.rack_tier_step_n - 0.025),
        ));

        // Side-of-rack pneumatic Cap-Swap Station bolted onto the right (+t_rack) upright
        spawn_cap_swap_station(
            parent,
            meshes,
            materials,
            studio,
            max_cans,
            half_w,
            rack_steel_mat,
        );
    });

    // Subtle slot number verification: ensure every slot from StudioConfig sits above a shelf
    debug_assert_eq!(slots.len(), max_cans);
}

#[inline]
fn cap_dock_nozzle_world(dock: &crate::core::types::CapDockPose) -> Vec3 {
    dock.grasp_pos + dock.grasp_rot * Vec3::new(0.0, 0.085, -0.015)
}

fn spawn_cap_swap_station(
    parent: &mut ChildSpawnerCommands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    studio: &StudioConfig,
    max_cans: usize,
    rack_half_w: f32,
    gunmetal_mat: Handle<StandardMaterial>,
) {
    let cap_docks = studio.cap_station_poses(max_cans);
    let rack_origin = Vec3::new(studio.rack_center_xz.x, 0.0, studio.rack_center_xz.z);
    let rack_rot_inv = Quat::from_rotation_y(studio.rack_yaw_deg.to_radians()).inverse();

    let dock0_nozzle = rack_rot_inv * (cap_dock_nozzle_world(&cap_docks[0]) - rack_origin);
    let dock1_nozzle = rack_rot_inv * (cap_dock_nozzle_world(&cap_docks[1]) - rack_origin);

    let aluminum_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.79, 0.83),
        metallic: 0.90,
        perceptual_roughness: 0.24,
        ..default()
    });
    let safety_orange_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#E85D1A").unwrap().into(),
        metallic: 0.14,
        perceptual_roughness: 0.34,
        ..default()
    });
    let dark_socket_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.07, 0.08, 0.09),
        metallic: 0.45,
        perceptual_roughness: 0.55,
        ..default()
    });
    let white_cap_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.95, 0.94),
        metallic: 0.02,
        perceptual_roughness: 0.38,
        ..default()
    });
    let cyan_cap_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#00E5FF").unwrap().into(),
        metallic: 0.12,
        perceptual_roughness: 0.32,
        ..default()
    });
    let brass_fitting_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.80, 0.64, 0.28),
        metallic: 0.88,
        perceptual_roughness: 0.28,
        ..default()
    });

    // 1. Welded gunmetal cantilever bracket bolted onto the right (+t_rack) rear upright
    let x_upright_outer = rack_half_w + 0.006;
    let x_station_outer = dock1_nozzle.x + 0.042;
    let beam_w = (x_station_outer - x_upright_outer).max(0.18);
    let beam_cx = 0.5 * (x_upright_outer + x_station_outer);
    let rear_z = -studio.rack_tier_step_n - 0.025;

    // Upright clamp flange plate
    parent.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.012, 0.145, 0.062))),
        MeshMaterial3d(gunmetal_mat.clone()),
        Transform::from_xyz(x_upright_outer + 0.004, 1.078, rear_z + 0.006),
    ));
    // Horizontal cantilevered steel box beam
    parent.spawn((
        Mesh3d(meshes.add(Cuboid::new(beam_w, 0.026, 0.040))),
        MeshMaterial3d(gunmetal_mat.clone()),
        Transform::from_xyz(beam_cx, 1.084, -0.044),
    ));
    // Angled welded diagonal gusset strut beneath the cantilever beam
    parent.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.145, 0.018, 0.024))),
        MeshMaterial3d(gunmetal_mat.clone()),
        Transform::from_xyz(x_upright_outer + 0.068, 1.036, -0.046)
            .with_rotation(Quat::from_rotation_z(0.58)),
    ));
    // Outboard end plate
    parent.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.008, 0.056, 0.068))),
        MeshMaterial3d(gunmetal_mat.clone()),
        Transform::from_xyz(x_station_outer + 0.004, 1.110, -0.024),
    ));

    // 2. Brushed aluminum pneumatic manifold block (kept strictly at z <= +0.012 so can body at z >= +0.027 clears)
    let manifold_x0 = (dock0_nozzle.x - 0.068).max(x_upright_outer + 0.012);
    let manifold_x1 = x_station_outer;
    let manifold_w = (manifold_x1 - manifold_x0).max(0.16);
    let manifold_cx = 0.5 * (manifold_x0 + manifold_x1);

    parent.spawn((
        Mesh3d(meshes.add(Cuboid::new(manifold_w, 0.042, 0.060))),
        MeshMaterial3d(aluminum_mat.clone()),
        Transform::from_xyz(manifold_cx, 1.116, -0.020),
    ));

    // Brass pneumatic air-line fittings on rear face of manifold
    let fitting_mesh = meshes.add(Cylinder::new(0.0055, 0.014));
    for dx in [-0.040_f32, 0.040] {
        parent.spawn((
            Mesh3d(fitting_mesh.clone()),
            MeshMaterial3d(brass_fitting_mat.clone()),
            Transform::from_xyz(manifold_cx + dx, 1.116, -0.056)
                .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        ));
    }

    // 3. Safety-orange front fascia bar across manifold front (+n_rack face, z <= +0.018)
    parent.spawn((
        Mesh3d(meshes.add(Cuboid::new(manifold_w + 0.006, 0.018, 0.008))),
        MeshMaterial3d(safety_orange_mat),
        Transform::from_xyz(manifold_cx, 1.114, 0.014),
    ));

    // 4. Pneumatic cap-extractor collet cups + color-coded ring collars + spare cap magazines
    let collet_arm_mesh = meshes.add(Cuboid::new(0.016, 0.008, 0.038));
    let collet_cup_mesh = meshes.add(Cylinder::new(0.0095, 0.009));
    let collet_bore_mesh = meshes.add(Cylinder::new(0.0062, 0.002));
    let collar_ring_mesh = meshes.add(Cylinder::new(0.0118, 0.003));
    let spare_socket_mesh = meshes.add(Cylinder::new(0.0096, 0.003));
    let spare_fat_body_mesh = meshes.add(Cylinder::new(0.0082, 0.015));
    let spare_fat_tip_mesh = meshes.add(Cylinder::new(0.0028, 0.006));
    let spare_skinny_body_mesh = meshes.add(Cylinder::new(0.0078, 0.016));
    let spare_skinny_needle_mesh = meshes.add(Cylinder::new(0.0014, 0.008));

    let dock_specs = [
        (dock0_nozzle, white_cap_mat.clone(), true),
        (dock1_nozzle, cyan_cap_mat.clone(), false),
    ];

    for (nozzle_local, accent_mat, is_fat) in dock_specs {
        let y_cup = nozzle_local.y - 0.004;

        // Slim cantilevered extractor arm from manifold front (z = 0.008) to nozzle tip (z = nozzle_local.z)
        let arm_z_start = 0.008;
        let arm_z_mid = 0.5 * (arm_z_start + nozzle_local.z);
        parent.spawn((
            Mesh3d(collet_arm_mesh.clone()),
            MeshMaterial3d(aluminum_mat.clone()),
            Transform::from_xyz(nozzle_local.x, y_cup - 0.005, arm_z_mid),
        ));

        // Pneumatic cap-extractor collet cup cleanly below/around nozzle tip with open top socket
        parent.spawn((
            Mesh3d(collet_cup_mesh.clone()),
            MeshMaterial3d(gunmetal_mat.clone()),
            Transform::from_xyz(nozzle_local.x, y_cup, nozzle_local.z),
        ));
        parent.spawn((
            Mesh3d(collet_bore_mesh.clone()),
            MeshMaterial3d(dark_socket_mat.clone()),
            Transform::from_xyz(nozzle_local.x, y_cup + 0.0042, nozzle_local.z),
        ));

        // Color-coded identification ring collar around the dock collet cup
        parent.spawn((
            Mesh3d(collar_ring_mesh.clone()),
            MeshMaterial3d(accent_mat.clone()),
            Transform::from_xyz(nozzle_local.x, y_cup + 0.0015, nozzle_local.z),
        ));

        // 3 spare caps arranged neatly on the manifold rear magazine shelf beside/behind each dock
        let magazine_y = 1.1385;
        let magazine_z = -0.024;
        for offset_x in [-0.020_f32, 0.0, 0.020] {
            let cx = nozzle_local.x + offset_x;
            parent.spawn((
                Mesh3d(spare_socket_mesh.clone()),
                MeshMaterial3d(dark_socket_mat.clone()),
                Transform::from_xyz(cx, magazine_y, magazine_z),
            ));
            if is_fat {
                parent.spawn((
                    Mesh3d(spare_fat_body_mesh.clone()),
                    MeshMaterial3d(accent_mat.clone()),
                    Transform::from_xyz(cx, magazine_y + 0.0085, magazine_z),
                ));
                parent.spawn((
                    Mesh3d(spare_fat_tip_mesh.clone()),
                    MeshMaterial3d(dark_socket_mat.clone()),
                    Transform::from_xyz(cx, magazine_y + 0.011, magazine_z - 0.007)
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                ));
            } else {
                parent.spawn((
                    Mesh3d(spare_skinny_body_mesh.clone()),
                    MeshMaterial3d(accent_mat.clone()),
                    Transform::from_xyz(cx, magazine_y + 0.009, magazine_z),
                ));
                parent.spawn((
                    Mesh3d(spare_skinny_needle_mesh.clone()),
                    MeshMaterial3d(aluminum_mat.clone()),
                    Transform::from_xyz(cx, magazine_y + 0.011, magazine_z - 0.008)
                        .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                ));
            }
        }
    }
}

fn build_wrinkled_drop_cloth_mesh(width: f32, depth: f32, cols: usize, rows: usize) -> Mesh {
    let cols = cols.max(2);
    let rows = rows.max(2);
    let nx = cols + 1;
    let nz = rows + 1;
    let total = nx * nz;

    let mut heights = vec![0.0_f32; total];
    for iz in 0..nz {
        for ix in 0..nx {
            let u = ix as f32 / cols as f32;
            let v = iz as f32 / rows as f32;
            let edge_u = u.min(1.0 - u) * width;
            let edge_v = v.min(1.0 - v) * depth;
            // Taped perimeter border (edge <= 0.038m) lies flat on the concrete floor
            let fade_u = ((edge_u - 0.038) / 0.07).clamp(0.0, 1.0);
            let fade_v = ((edge_v - 0.038) / 0.07).clamp(0.0, 1.0);
            let mask = fade_u * fade_v;

            let crease_x = (fbm(u * 9.0, v * 3.5, 401) - 0.5).abs();
            let crease_ridge = (1.0 - (crease_x * 9.0).clamp(0.0, 1.0)).powi(3);
            let fold = ((u * 21.0 + v * 3.0 * PI).sin() * 0.5 + 0.5) * 0.35;
            heights[iz * nx + ix] = mask * (0.0046 * crease_ridge + 0.0024 * fold);
        }
    }

    let dx_step = width / cols as f32;
    let dz_step = depth / rows as f32;
    let mut positions = Vec::with_capacity(total);
    let mut normals = Vec::with_capacity(total);
    let mut uvs = Vec::with_capacity(total);

    for iz in 0..nz {
        for ix in 0..nx {
            let u = ix as f32 / cols as f32;
            let v = iz as f32 / rows as f32;
            let px = (u - 0.5) * width;
            let pz = (v - 0.5) * depth;
            let py = heights[iz * nx + ix];
            positions.push([px, py, pz]);
            uvs.push([u, v]);

            let xm = ix.saturating_sub(1);
            let xp = (ix + 1).min(cols);
            let zm = iz.saturating_sub(1);
            let zp = (iz + 1).min(rows);
            let dhdx = (heights[iz * nx + xp] - heights[iz * nx + xm])
                / ((xp - xm).max(1) as f32 * dx_step);
            let dhdz = (heights[zp * nx + ix] - heights[zm * nx + ix])
                / ((zp - zm).max(1) as f32 * dz_step);
            let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
            normals.push([n.x, n.y, n.z]);
        }
    }

    let mut indices = Vec::with_capacity(cols * rows * 6);
    for iz in 0..rows as u32 {
        for ix in 0..cols as u32 {
            let a = iz * nx as u32 + ix;
            let b = (iz + 1) * nx as u32 + ix;
            let c = iz * nx as u32 + ix + 1;
            let d = (iz + 1) * nx as u32 + ix + 1;
            indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
        .with_generated_tangents()
        .expect("drop cloth tangents")
}

fn spawn_drop_cloth_runner(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
) {
    let (dc_albedo, dc_rough, dc_normal) = build_drop_cloth_textures(1024, 256);
    let dc_mat = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(dc_albedo)),
        metallic_roughness_texture: Some(images.add(dc_rough)),
        normal_map_texture: Some(images.add(dc_normal)),
        metallic: 0.0,
        perceptual_roughness: 1.0,
        reflectance: 0.25,
        ..default()
    });
    // Spans X in [-1.42, +1.42], Z in [0.04, 0.72] — clears rail drag-chain (Z >= 0.7505) & rack feet (X <= -1.514)
    let dc_mesh = meshes.add(build_wrinkled_drop_cloth_mesh(2.84, 0.68, 48, 14));
    commands.spawn((
        Mesh3d(dc_mesh),
        MeshMaterial3d(dc_mat),
        Transform::from_xyz(0.0, 0.0022, 0.38),
        NotShadowCaster,
    ));
}

fn spawn_overhead_shop_lights(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    steel_struct_mat: &Handle<StandardMaterial>,
) {
    let reflector_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.22, 0.24, 0.27),
        metallic: 0.78,
        perceptual_roughness: 0.36,
        ..default()
    });
    let tube_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.96, 0.98, 1.0),
        emissive: LinearRgba::new(4.2, 4.5, 5.2, 1.0),
        unlit: true,
        ..default()
    });

    let hood_mesh = meshes.add(Cuboid::new(1.24, 0.036, 0.17));
    let cap_mesh = meshes.add(Cuboid::new(0.02, 0.062, 0.176));
    let rod_mesh = meshes.add(Cylinder::new(0.006, 0.16));
    let tube_mesh = meshes.add(Cylinder::new(0.015, 1.18));

    // 6 linear LED/fluorescent fixtures hung at y ≈ 3.94m along the trusses (out of Overview/Canvas/Follow sightlines)
    for &tz in &[0.55_f32, 2.20, 3.85] {
        for &fx in &[-1.45_f32, 1.45] {
            commands.spawn((
                Transform::from_xyz(fx, 3.94, tz),
                Visibility::default(),
                children![
                    // Steel reflector hood
                    (
                        Mesh3d(hood_mesh.clone()),
                        MeshMaterial3d(reflector_mat.clone()),
                        Transform::from_xyz(0.0, 0.018, 0.0),
                        NotShadowCaster,
                    ),
                    // Left & right end-caps
                    (
                        Mesh3d(cap_mesh.clone()),
                        MeshMaterial3d(steel_struct_mat.clone()),
                        Transform::from_xyz(-0.615, 0.005, 0.0),
                        NotShadowCaster,
                    ),
                    (
                        Mesh3d(cap_mesh.clone()),
                        MeshMaterial3d(steel_struct_mat.clone()),
                        Transform::from_xyz(0.615, 0.005, 0.0),
                        NotShadowCaster,
                    ),
                    // Hanger rods up to top chord (y = 4.12m)
                    (
                        Mesh3d(rod_mesh.clone()),
                        MeshMaterial3d(steel_struct_mat.clone()),
                        Transform::from_xyz(-0.46, 0.10, 0.0),
                        NotShadowCaster,
                    ),
                    (
                        Mesh3d(rod_mesh.clone()),
                        MeshMaterial3d(steel_struct_mat.clone()),
                        Transform::from_xyz(0.46, 0.10, 0.0),
                        NotShadowCaster,
                    ),
                    // Twin cool-white fluorescent/LED tubes
                    (
                        Mesh3d(tube_mesh.clone()),
                        MeshMaterial3d(tube_mat.clone()),
                        Transform::from_xyz(0.0, -0.014, -0.042)
                            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
                        NotShadowCaster,
                    ),
                    (
                        Mesh3d(tube_mesh.clone()),
                        MeshMaterial3d(tube_mat.clone()),
                        Transform::from_xyz(0.0, -0.014, 0.042)
                            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
                        NotShadowCaster,
                    ),
                ],
            ));
        }
    }
}

fn spawn_perimeter_workshop_props(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    steel_struct_mat: &Handle<StandardMaterial>,
) {
    let conduit_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.65, 0.69),
        metallic: 0.86,
        perceptual_roughness: 0.34,
        ..default()
    });
    let panel_grey_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.35, 0.39),
        metallic: 0.55,
        perceptual_roughness: 0.48,
        ..default()
    });
    let safety_yellow_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#D99B16").unwrap().into(),
        metallic: 0.18,
        perceptual_roughness: 0.42,
        ..default()
    });
    let estop_red_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#C92420").unwrap().into(),
        metallic: 0.08,
        perceptual_roughness: 0.36,
        ..default()
    });
    let status_green_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.15, 0.85, 0.32),
        emissive: LinearRgba::new(0.25, 1.4, 0.45, 1.0),
        ..default()
    });
    let compressor_blue_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#234E70").unwrap().into(),
        metallic: 0.42,
        perceptual_roughness: 0.38,
        ..default()
    });
    let brass_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.62, 0.26),
        metallic: 0.88,
        perceptual_roughness: 0.30,
        ..default()
    });
    let cardboard_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.58, 0.44, 0.30),
        metallic: 0.0,
        perceptual_roughness: 0.88,
        ..default()
    });
    let tape_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.60, 0.42),
        metallic: 0.0,
        perceptual_roughness: 0.46,
        ..default()
    });
    let hose_orange_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#D45B21").unwrap().into(),
        metallic: 0.04,
        perceptual_roughness: 0.56,
        ..default()
    });
    let dark_rubber_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.085, 0.095),
        metallic: 0.05,
        perceptual_roughness: 0.78,
        ..default()
    });

    // 1. Back-wall electrical conduit pipes, breaker/junction boxes & pneumatic FRL regulator (strictly |x| > 2.25m)
    let vert_conduit_mesh = meshes.add(Cylinder::new(0.013, 3.75));
    let clamp_mesh = meshes.add(Cuboid::new(0.044, 0.018, 0.022));
    for &cx in &[2.38_f32, 2.95, -3.15] {
        commands.spawn((
            Mesh3d(vert_conduit_mesh.clone()),
            MeshMaterial3d(conduit_mat.clone()),
            Transform::from_xyz(cx, 2.00, -0.025),
        ));
        for &cy in &[0.65_f32, 1.55, 2.45, 3.35] {
            commands.spawn((
                Mesh3d(clamp_mesh.clone()),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(cx, cy, -0.028),
            ));
        }
    }
    // Horizontal right-back conduit run & left-back short crossover
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.012, 1.30))),
        MeshMaterial3d(conduit_mat.clone()),
        Transform::from_xyz(3.03, 2.15, -0.025).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.012, 0.40))),
        MeshMaterial3d(conduit_mat.clone()),
        Transform::from_xyz(-2.95, 1.50, -0.025).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    ));

    // Right NEMA steel electrical panel / junction box (X = 2.66m)
    commands.spawn((
        Transform::from_xyz(2.66, 1.65, 0.016),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cuboid::new(0.36, 0.48, 0.11))),
                MeshMaterial3d(panel_grey_mat.clone()),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.32, 0.44, 0.014))),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(0.0, 0.0, 0.054),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.018, 0.06, 0.02))),
                MeshMaterial3d(conduit_mat.clone()),
                Transform::from_xyz(-0.12, 0.0, 0.065),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.008, 0.012))),
                MeshMaterial3d(status_green_mat.clone()),
                Transform::from_xyz(0.10, 0.16, 0.062)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
        ],
    ));

    // Right pneumatic air-regulator box & pressure dial (X = 3.20m)
    commands.spawn((
        Transform::from_xyz(3.20, 1.46, 0.008),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cuboid::new(0.22, 0.26, 0.09))),
                MeshMaterial3d(panel_grey_mat.clone()),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.032, 0.024))),
                MeshMaterial3d(brass_mat.clone()),
                Transform::from_xyz(0.0, 0.03, 0.052)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.014, 0.16))),
                MeshMaterial3d(hose_orange_mat.clone()),
                Transform::from_xyz(0.0, -0.20, 0.02),
            ),
        ],
    ));

    // Left back-wall junction box & pneumatic shutoff block (X = -3.15, -2.75)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.20, 0.24, 0.08))),
        MeshMaterial3d(panel_grey_mat.clone()),
        Transform::from_xyz(-3.15, 2.25, -0.002),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.16, 0.18, 0.07))),
        MeshMaterial3d(panel_grey_mat.clone()),
        Transform::from_xyz(-2.75, 1.50, -0.005),
    ));

    // 2. Floor rail right-end cable junction box & Safety/E-Stop pedestal (strictly outside X > 1.85 & robot sweep X <= 2.05)
    commands.spawn((
        Transform::from_xyz(1.90, 0.042, 0.78),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cuboid::new(0.18, 0.084, 0.14))),
                MeshMaterial3d(panel_grey_mat.clone()),
            ),
            // Armored floor cable conduit running to the right wall baseboard
            (
                Mesh3d(meshes.add(Cylinder::new(0.014, 2.04))),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(1.06, -0.028, 0.0)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
        ],
    ));

    // Safety / E-Stop pedestal beside right end of rail (X = 2.28m, Z = 1.08m)
    commands.spawn((
        Transform::from_xyz(2.28, 0.0, 1.08),
        Visibility::default(),
        children![
            // Base plate
            (
                Mesh3d(meshes.add(Cuboid::new(0.16, 0.016, 0.16))),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(0.0, 0.008, 0.0),
            ),
            // Yellow safety post
            (
                Mesh3d(meshes.add(Cuboid::new(0.055, 0.96, 0.055))),
                MeshMaterial3d(safety_yellow_mat.clone()),
                Transform::from_xyz(0.0, 0.49, 0.0),
            ),
            // Control head enclosure
            (
                Mesh3d(meshes.add(Cuboid::new(0.13, 0.11, 0.13))),
                MeshMaterial3d(panel_grey_mat.clone()),
                Transform::from_xyz(0.0, 1.01, 0.0),
            ),
            // Yellow E-Stop collar & red mushroom button
            (
                Mesh3d(meshes.add(Cylinder::new(0.036, 0.012))),
                MeshMaterial3d(safety_yellow_mat.clone()),
                Transform::from_xyz(-0.015, 1.07, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.024, 0.028))),
                MeshMaterial3d(estop_red_mat),
                Transform::from_xyz(-0.015, 1.086, 0.0),
            ),
            // Green reset indicator button
            (
                Mesh3d(meshes.add(Cylinder::new(0.012, 0.012))),
                MeshMaterial3d(status_green_mat),
                Transform::from_xyz(0.035, 1.069, 0.025),
            ),
        ],
    ));

    // 3. Industrial air compressor tank tucked against far left side wall (X = -3.45m, Z = 1.68m)
    commands.spawn((
        Transform::from_xyz(-3.45, 0.0, 1.68),
        Visibility::default(),
        children![
            // Saddle feet
            (
                Mesh3d(meshes.add(Cuboid::new(0.32, 0.09, 0.08))),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(0.0, 0.045, -0.24),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.32, 0.09, 0.08))),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(0.0, 0.045, 0.24),
            ),
            // Horizontal cylindrical pressure tank
            (
                Mesh3d(meshes.add(Cylinder::new(0.19, 0.78))),
                MeshMaterial3d(compressor_blue_mat),
                Transform::from_xyz(0.0, 0.27, 0.0)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
            // Motor & pump block on top saddle
            (
                Mesh3d(meshes.add(Cuboid::new(0.24, 0.022, 0.34))),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_xyz(0.0, 0.46, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cylinder::new(0.085, 0.22))),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(0.0, 0.56, -0.05)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.14, 0.17, 0.14))),
                MeshMaterial3d(conduit_mat.clone()),
                Transform::from_xyz(0.0, 0.56, 0.12),
            ),
            // Brass regulator valve + gauge
            (
                Mesh3d(meshes.add(Cylinder::new(0.022, 0.03))),
                MeshMaterial3d(brass_mat),
                Transform::from_xyz(0.14, 0.49, 0.16)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
        ],
    ));

    // 4. Taped corrugated cardboard spray-can supply cartons tucked against far side walls
    spawn_taped_carton(
        commands,
        meshes,
        &cardboard_mat,
        &tape_mat,
        Vec3::new(-3.52, 0.0, 2.52),
        Vec3::new(0.42, 0.32, 0.48),
        0.08,
    );
    spawn_taped_carton(
        commands,
        meshes,
        &cardboard_mat,
        &tape_mat,
        Vec3::new(-3.50, 0.32, 2.54),
        Vec3::new(0.36, 0.26, 0.38),
        -0.14,
    );
    spawn_taped_carton(
        commands,
        meshes,
        &cardboard_mat,
        &tape_mat,
        Vec3::new(3.48, 0.0, 0.52),
        Vec3::new(0.40, 0.30, 0.44),
        -0.10,
    );

    // 5. Galvanized overhead spray-booth exhaust duct & extraction baffle hood (Y >= 2.95m, safely above canvas 2.30m)
    let duct_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.66, 0.69, 0.73),
        metallic: 0.88,
        perceptual_roughness: 0.30,
        clearcoat: 0.25,
        clearcoat_perceptual_roughness: 0.28,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.15, 7.60))),
        MeshMaterial3d(duct_mat.clone()),
        Transform::from_xyz(0.0, 3.42, 0.22).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    ));
    let duct_ring_mesh = meshes.add(Cylinder::new(0.162, 0.028));
    let strap_mesh = meshes.add(Cuboid::new(0.028, 0.78, 0.012));
    for ix in -3..=3 {
        let rx = ix as f32 * 1.15;
        commands.spawn((
            Mesh3d(duct_ring_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(rx, 3.42, 0.22).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        ));
        commands.spawn((
            Mesh3d(strap_mesh.clone()),
            MeshMaterial3d(steel_struct_mat.clone()),
            Transform::from_xyz(rx, 3.81, 0.22),
        ));
    }
    // Baffle hood above mural centered at (0, 2.95, 0.16) + vertical collar up into duct
    commands.spawn((
        Transform::from_xyz(0.0, 2.95, 0.16),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cuboid::new(2.60, 0.045, 0.28))),
                MeshMaterial3d(steel_struct_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_x(-0.22)),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(2.50, 0.018, 0.22))),
                MeshMaterial3d(duct_mat.clone()),
                Transform::from_xyz(0.0, 0.030, -0.02)
                    .with_rotation(Quat::from_rotation_x(-0.22)),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.54, 0.44, 0.18))),
                MeshMaterial3d(duct_mat.clone()),
                Transform::from_xyz(-0.75, 0.24, 0.04),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.54, 0.44, 0.18))),
                MeshMaterial3d(duct_mat.clone()),
                Transform::from_xyz(0.75, 0.24, 0.04),
            ),
        ],
    ));

    // 6. Rolling industrial paint cart in left mid-ground (-2.95, 0.0, 0.68), yaw +0.22 rad (visible in Overview, clear of rack X >= -2.35)
    let cart_red_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#8E241C").unwrap().into(),
        metallic: 0.42,
        perceptual_roughness: 0.36,
        clearcoat: 0.48,
        clearcoat_perceptual_roughness: 0.20,
        ..default()
    });
    let rag_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#8A9BAE").unwrap().into(),
        metallic: 0.0,
        perceptual_roughness: 0.90,
        ..default()
    });
    let pendant_screen_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#2892B8").unwrap().into(),
        emissive: LinearRgba::new(0.18, 0.62, 0.85, 1.0),
        unlit: true,
        ..default()
    });
    let blue_tape_mat = materials.add(StandardMaterial {
        base_color: Srgba::hex("#2063A8").unwrap().into(),
        metallic: 0.02,
        perceptual_roughness: 0.50,
        ..default()
    });
    let caster_wheel = meshes.add(Cylinder::new(0.038, 0.024));
    let leg_post = meshes.add(Cuboid::new(0.032, 0.72, 0.032));
    let shelf_tray = meshes.add(Cuboid::new(0.64, 0.042, 0.42));
    let can_cyl = meshes.add(Cylinder::new(0.0328, 0.120));
    let can_label_cyl = meshes.add(Cylinder::new(0.0332, 0.062));
    let can_bottom_rim = meshes.add(Cylinder::new(0.0330, 0.010));
    let can_shoulder = meshes.add(
        ConicalFrustum {
            radius_top: 0.018,
            radius_bottom: 0.0328,
            height: 0.024,
        }
        .mesh()
        .resolution(24),
    );
    let can_cap = meshes.add(Cylinder::new(0.020, 0.022));
    let can_nozzle = meshes.add(Cylinder::new(0.0085, 0.016));
    let cart_can_steel_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.79, 0.83),
        metallic: 0.90,
        perceptual_roughness: 0.22,
        ..default()
    });
    let cart_can_label_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.11, 0.12, 0.14),
        metallic: 0.04,
        perceptual_roughness: 0.52,
        ..default()
    });
    let cart_nozzle_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.92, 0.91),
        metallic: 0.02,
        perceptual_roughness: 0.42,
        ..default()
    });
    let can_colors = [
        Color::srgb(0.05, 0.72, 0.88),
        Color::srgb(0.92, 0.38, 0.12),
        Color::srgb(0.86, 0.18, 0.44),
        Color::srgb(0.92, 0.74, 0.16),
    ];
    let can_mats: Vec<Handle<StandardMaterial>> = can_colors
        .iter()
        .map(|&col| {
            materials.add(StandardMaterial {
                base_color: col,
                metallic: 0.12,
                perceptual_roughness: 0.32,
                clearcoat: 0.65,
                clearcoat_perceptual_roughness: 0.18,
                ..default()
            })
        })
        .collect();
    let cart_can_transforms = [
        Transform::from_xyz(-0.18, 0.896, -0.06),
        Transform::from_xyz(-0.10, 0.896, -0.08),
        Transform::from_xyz(-0.14, 0.338, 0.02),
        Transform::from_xyz(0.02, 0.852, 0.08)
            .with_rotation(Quat::from_euler(EulerRot::YXZ, 0.48, FRAC_PI_2, 0.0)),
    ];

    commands.spawn((
        Transform::from_xyz(-2.95, 0.0, 0.68).with_rotation(Quat::from_rotation_y(0.22)),
        Visibility::default(),
        children![
            // 4 swivel casters
            (
                Mesh3d(caster_wheel.clone()),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(-0.27, 0.038, -0.16).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            (
                Mesh3d(caster_wheel.clone()),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(0.27, 0.038, -0.16).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            (
                Mesh3d(caster_wheel.clone()),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(-0.27, 0.038, 0.16).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            (
                Mesh3d(caster_wheel),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(0.27, 0.038, 0.16).with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ),
            // 4 upright angle legs
            (
                Mesh3d(leg_post.clone()),
                MeshMaterial3d(cart_red_mat.clone()),
                Transform::from_xyz(-0.29, 0.43, -0.18),
            ),
            (
                Mesh3d(leg_post.clone()),
                MeshMaterial3d(cart_red_mat.clone()),
                Transform::from_xyz(0.29, 0.43, -0.18),
            ),
            (
                Mesh3d(leg_post.clone()),
                MeshMaterial3d(cart_red_mat.clone()),
                Transform::from_xyz(-0.29, 0.43, 0.18),
            ),
            (
                Mesh3d(leg_post),
                MeshMaterial3d(cart_red_mat.clone()),
                Transform::from_xyz(0.29, 0.43, 0.18),
            ),
            // Lower & upper steel trays + rubber mat on upper deck
            (
                Mesh3d(shelf_tray.clone()),
                MeshMaterial3d(cart_red_mat.clone()),
                Transform::from_xyz(0.0, 0.24, 0.0),
            ),
            (
                Mesh3d(shelf_tray),
                MeshMaterial3d(cart_red_mat),
                Transform::from_xyz(0.0, 0.79, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.58, 0.008, 0.36))),
                MeshMaterial3d(dark_rubber_mat.clone()),
                Transform::from_xyz(0.0, 0.815, 0.0),
            ),
            // Tubular push handle on right edge (+X)
            (
                Mesh3d(meshes.add(Cylinder::new(0.013, 0.40))),
                MeshMaterial3d(conduit_mat.clone()),
                Transform::from_xyz(0.36, 0.84, 0.0).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ),
            // 3 upright spare spray cans + 1 tipped on its side on top tray
            (
                cart_can_transforms[0],
                Visibility::default(),
                children![
                    (Mesh3d(can_bottom_rim.clone()), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, -0.072, 0.0)),
                    (Mesh3d(can_cyl.clone()), MeshMaterial3d(can_mats[0].clone()), Transform::from_xyz(0.0, -0.012, 0.0)),
                    (Mesh3d(can_label_cyl.clone()), MeshMaterial3d(cart_can_label_mat.clone()), Transform::from_xyz(0.0, -0.016, 0.0)),
                    (Mesh3d(can_shoulder.clone()), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, 0.060, 0.0)),
                    (Mesh3d(can_cap.clone()), MeshMaterial3d(can_mats[0].clone()), Transform::from_xyz(0.0, 0.078, 0.0)),
                    (Mesh3d(can_nozzle.clone()), MeshMaterial3d(cart_nozzle_mat.clone()), Transform::from_xyz(0.0, 0.092, 0.0)),
                ],
            ),
            (
                cart_can_transforms[1],
                Visibility::default(),
                children![
                    (Mesh3d(can_bottom_rim.clone()), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, -0.072, 0.0)),
                    (Mesh3d(can_cyl.clone()), MeshMaterial3d(can_mats[1].clone()), Transform::from_xyz(0.0, -0.012, 0.0)),
                    (Mesh3d(can_label_cyl.clone()), MeshMaterial3d(cart_can_label_mat.clone()), Transform::from_xyz(0.0, -0.016, 0.0)),
                    (Mesh3d(can_shoulder.clone()), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, 0.060, 0.0)),
                    (Mesh3d(can_cap.clone()), MeshMaterial3d(can_mats[1].clone()), Transform::from_xyz(0.0, 0.078, 0.0)),
                    (Mesh3d(can_nozzle.clone()), MeshMaterial3d(cart_nozzle_mat.clone()), Transform::from_xyz(0.0, 0.092, 0.0)),
                ],
            ),
            (
                cart_can_transforms[2],
                Visibility::default(),
                children![
                    (Mesh3d(can_bottom_rim.clone()), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, -0.072, 0.0)),
                    (Mesh3d(can_cyl.clone()), MeshMaterial3d(can_mats[2].clone()), Transform::from_xyz(0.0, -0.012, 0.0)),
                    (Mesh3d(can_label_cyl.clone()), MeshMaterial3d(cart_can_label_mat.clone()), Transform::from_xyz(0.0, -0.016, 0.0)),
                    (Mesh3d(can_shoulder.clone()), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, 0.060, 0.0)),
                    (Mesh3d(can_cap.clone()), MeshMaterial3d(can_mats[2].clone()), Transform::from_xyz(0.0, 0.078, 0.0)),
                    (Mesh3d(can_nozzle.clone()), MeshMaterial3d(cart_nozzle_mat.clone()), Transform::from_xyz(0.0, 0.092, 0.0)),
                ],
            ),
            (
                cart_can_transforms[3],
                Visibility::default(),
                children![
                    (Mesh3d(can_bottom_rim), MeshMaterial3d(cart_can_steel_mat.clone()), Transform::from_xyz(0.0, -0.072, 0.0)),
                    (Mesh3d(can_cyl), MeshMaterial3d(can_mats[3].clone()), Transform::from_xyz(0.0, -0.012, 0.0)),
                    (Mesh3d(can_label_cyl), MeshMaterial3d(cart_can_label_mat), Transform::from_xyz(0.0, -0.016, 0.0)),
                    (Mesh3d(can_shoulder), MeshMaterial3d(cart_can_steel_mat), Transform::from_xyz(0.0, 0.060, 0.0)),
                    (Mesh3d(can_cap), MeshMaterial3d(can_mats[3].clone()), Transform::from_xyz(0.0, 0.078, 0.0)),
                    (Mesh3d(can_nozzle), MeshMaterial3d(cart_nozzle_mat), Transform::from_xyz(0.0, 0.092, 0.0)),
                ],
            ),
            // Roll of blue painter's tape
            (
                Mesh3d(meshes.add(
                    Torus {
                        minor_radius: 0.016,
                        major_radius: 0.044,
                    }
                    .mesh()
                    .minor_resolution(12)
                    .major_resolution(24)
                    .build(),
                )),
                MeshMaterial3d(blue_tape_mat),
                Transform::from_xyz(0.16, 0.835, -0.07),
            ),
            // Folded shop rag
            (
                Mesh3d(meshes.add(Cuboid::new(0.16, 0.018, 0.12))),
                MeshMaterial3d(rag_mat),
                Transform::from_xyz(0.18, 0.828, 0.08).with_rotation(Quat::from_rotation_y(-0.35)),
            ),
            // Handheld teach pendant + coiled cord on front shelf edge
            (
                Mesh3d(meshes.add(Cuboid::new(0.11, 0.024, 0.16))),
                MeshMaterial3d(panel_grey_mat.clone()),
                Transform::from_xyz(-0.02, 0.830, -0.10).with_rotation(Quat::from_rotation_y(0.18)),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.082, 0.004, 0.068))),
                MeshMaterial3d(pendant_screen_mat),
                Transform::from_xyz(-0.02, 0.842, -0.105).with_rotation(Quat::from_rotation_y(0.18)),
            ),
        ],
    ));

    // 7. Draped static pneumatic supply hose routed cleanly behind cart & rack along left/back wall
    //    and armored cable feed (right rail junction -> rail drag-chain anchor)
    commands.spawn((
        Mesh3d(meshes.add(build_static_hose_tube_mesh(
            Vec3::new(-3.31, 0.49, 1.84),
            Vec3::new(0.05, -0.55, -0.75),
            Vec3::new(-3.72, 0.08, 0.08),
            Vec3::new(-0.15, -0.10, -1.20),
            0.0095,
            18,
        ))),
        MeshMaterial3d(hose_orange_mat.clone()),
    ));
    commands.spawn((
        Mesh3d(meshes.add(build_static_hose_tube_mesh(
            Vec3::new(-3.72, 0.08, 0.08),
            Vec3::new(0.75, 0.15, -0.15),
            Vec3::new(-2.75, 1.41, -0.005),
            Vec3::new(0.10, 1.15, -0.05),
            0.0095,
            18,
        ))),
        MeshMaterial3d(hose_orange_mat),
    ));
    commands.spawn((
        Mesh3d(meshes.add(build_static_hose_tube_mesh(
            Vec3::new(1.81, 0.042, 0.78),
            Vec3::new(-0.18, 0.06, 0.0),
            Vec3::new(1.68, 0.065, 0.778),
            Vec3::new(-0.18, 0.0, 0.0),
            0.0115,
            10,
        ))),
        MeshMaterial3d(dark_rubber_mat),
    ));
}

fn spawn_taped_carton(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    cardboard_mat: &Handle<StandardMaterial>,
    tape_mat: &Handle<StandardMaterial>,
    base_pos: Vec3,
    size: Vec3,
    yaw_rad: f32,
) {
    let center = base_pos + Vec3::new(0.0, size.y * 0.5, 0.0);
    commands.spawn((
        Transform::from_translation(center).with_rotation(Quat::from_rotation_y(yaw_rad)),
        Visibility::default(),
        children![
            (
                Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
                MeshMaterial3d(cardboard_mat.clone()),
            ),
            // Packing tape strip across top seam & slightly down front/back flaps
            (
                Mesh3d(meshes.add(Cuboid::new(0.055, 0.003, size.z + 0.004))),
                MeshMaterial3d(tape_mat.clone()),
                Transform::from_xyz(0.0, size.y * 0.5 + 0.001, 0.0),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.055, 0.08, 0.003))),
                MeshMaterial3d(tape_mat.clone()),
                Transform::from_xyz(0.0, size.y * 0.5 - 0.038, size.z * 0.5 + 0.0015),
            ),
            (
                Mesh3d(meshes.add(Cuboid::new(0.055, 0.08, 0.003))),
                MeshMaterial3d(tape_mat.clone()),
                Transform::from_xyz(0.0, size.y * 0.5 - 0.038, -size.z * 0.5 - 0.0015),
            ),
        ],
    ));
}

fn spawn_studio_lights(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    images: &mut ResMut<Assets<Image>>,
) {
    // Balanced studio ambient bounce (54.0) for rich shadow depth & readable dark steel details
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.89, 0.92, 0.97),
        brightness: 54.0,
        affects_lightmapped_meshes: true,
    });

    // Subtle volumetric room haze volume (strictly crisp low density so canvas artwork never washes out)
    commands.spawn((
        FogVolume {
            fog_color: Color::srgb(0.88, 0.91, 0.96),
            density_factor: 0.010,
            absorption: 0.06,
            scattering: 0.25,
            scattering_asymmetry: 0.65,
            light_intensity: 0.85,
            ..default()
        },
        Transform::from_xyz(0.0, 2.35, 2.60).with_scale(Vec3::new(8.2, 4.4, 6.8)),
    ));

    // Cool overhead daylight skylight directional light casting crisp soft PCF shadows & subtle volumetric shafts
    commands.spawn((
        DirectionalLight {
            illuminance: 2_350.0,
            color: Color::srgb(0.95, 0.97, 1.0),
            shadow_maps_enabled: true,
            ..default()
        },
        VolumetricLight,
        Transform::from_xyz(2.0, 5.5, 4.2).looking_at(Vec3::new(-0.3, 1.2, 0.4), Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 5.0,
            maximum_distance: 15.0,
            ..default()
        }
        .build(),
    ));

    let housing_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.13, 0.15),
        metallic: 0.80,
        perceptual_roughness: 0.35,
        ..default()
    });
    let neutral_lens_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::new(5.5, 5.5, 5.5, 1.0),
        unlit: true,
        ..default()
    });
    let cool_lens_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.96, 1.0),
        emissive: LinearRgba::new(4.4, 5.0, 6.2, 1.0),
        unlit: true,
        ..default()
    });
    let warm_lens_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.85, 0.65),
        emissive: LinearRgba::new(6.8, 4.4, 2.2, 1.0),
        unlit: true,
        ..default()
    });

    let shaft_tex = images.add(build_light_shaft_texture(64, 128));
    let neutral_shaft_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.98, 0.95, 0.032),
        base_color_texture: Some(shaft_tex.clone()),
        alpha_mode: AlphaMode::Add,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let cool_shaft_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.84, 0.92, 1.0, 0.034),
        base_color_texture: Some(shaft_tex.clone()),
        alpha_mode: AlphaMode::Add,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let warm_shaft_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.78, 0.50, 0.042),
        base_color_texture: Some(shaft_tex),
        alpha_mode: AlphaMode::Add,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    let can_mesh = meshes.add(Cylinder::new(0.085, 0.22));
    let lens_mesh = meshes.add(Cylinder::new(0.072, 0.015));
    let shaft_mesh = meshes.add(make_open_cone_mesh(0.082, 0.56, 1.90, 32));

    // Spotlight 1: Neutral gallery key aimed cleanly at the wall canvas for accurate paint color
    commands.spawn((
        SpotLight {
            intensity: 175_000.0,
            color: Color::srgb(0.99, 0.99, 0.99),
            range: 12.0,
            inner_angle: 0.42,
            outer_angle: 0.72,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.25, 3.66, 2.45).looking_at(Vec3::new(0.0, 1.55, 0.0), Vec3::Y),
        children![
            (
                Mesh3d(can_mesh.clone()),
                MeshMaterial3d(housing_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(lens_mesh.clone()),
                MeshMaterial3d(neutral_lens_mat),
                Transform::from_xyz(0.0, 0.0, -0.105)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(shaft_mesh.clone()),
                MeshMaterial3d(neutral_shaft_mat),
                Transform::from_xyz(0.0, 0.0, -1.06)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
                NotShadowReceiver,
            ),
        ],
    ));

    // Spotlight 2: Cool overhead shop key aimed at the spray can rack and robot home station
    commands.spawn((
        SpotLight {
            intensity: 120_000.0,
            color: Color::srgb(0.86, 0.93, 1.0),
            range: 11.0,
            inner_angle: 0.36,
            outer_angle: 0.66,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-1.35, 3.66, 2.15).looking_at(Vec3::new(-1.60, 1.08, 0.58), Vec3::Y),
        children![
            (
                Mesh3d(can_mesh.clone()),
                MeshMaterial3d(housing_mat.clone()),
                Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(lens_mesh.clone()),
                MeshMaterial3d(cool_lens_mat),
                Transform::from_xyz(0.0, 0.0, -0.105)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(shaft_mesh.clone()),
                MeshMaterial3d(cool_shaft_mat),
                Transform::from_xyz(0.0, 0.0, -1.06)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
                NotShadowReceiver,
            ),
        ],
    ));

    // Spotlight 3: Warm tungsten/amber side work-light accent aimed at the robot/rail zone (tight cone keeps canvas neutral)
    commands.spawn((
        SpotLight {
            intensity: 110_000.0,
            color: Color::srgb(1.0, 0.77, 0.50),
            range: 9.5,
            inner_angle: 0.28,
            outer_angle: 0.52,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(2.55, 3.22, 2.30).looking_at(Vec3::new(-0.30, 1.02, 0.95), Vec3::Y),
        children![
            (
                Mesh3d(can_mesh),
                MeshMaterial3d(housing_mat),
                Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(lens_mesh),
                MeshMaterial3d(warm_lens_mat),
                Transform::from_xyz(0.0, 0.0, -0.105)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
            ),
            (
                Mesh3d(shaft_mesh),
                MeshMaterial3d(warm_shaft_mat),
                Transform::from_xyz(0.0, 0.0, -1.06)
                    .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
                NotShadowCaster,
                NotShadowReceiver,
            ),
        ],
    ));
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

#[derive(Clone, Copy)]
struct MistPatch {
    x: f32,
    z: f32,
    rx: f32,
    rz: f32,
    color: [f32; 3],
    alpha: f32,
    seed: u32,
}

const STUDIO_OVERSPRAY_PATCHES: [MistPatch; 10] = [
    // Subtle overspray clouds under canvas drip line (X in [-1.25, +1.25], Z in [0.05, 0.75])
    MistPatch { x: -0.88, z: 0.16, rx: 0.34, rz: 0.20, color: [0.10, 0.56, 0.82], alpha: 0.26, seed: 101 },
    MistPatch { x: -0.34, z: 0.24, rx: 0.36, rz: 0.24, color: [0.86, 0.34, 0.15], alpha: 0.23, seed: 202 },
    MistPatch { x: 0.18, z: 0.14, rx: 0.42, rz: 0.20, color: [0.12, 0.16, 0.20], alpha: 0.28, seed: 303 },
    MistPatch { x: 0.76, z: 0.22, rx: 0.34, rz: 0.22, color: [0.80, 0.20, 0.44], alpha: 0.22, seed: 404 },
    MistPatch { x: -0.10, z: 0.44, rx: 0.44, rz: 0.24, color: [0.84, 0.62, 0.16], alpha: 0.17, seed: 505 },
    MistPatch { x: 0.52, z: 0.48, rx: 0.32, rz: 0.22, color: [0.14, 0.68, 0.66], alpha: 0.18, seed: 606 },
    // Faint dried purge/overspray mist near spray-can rack & cap-swap station
    MistPatch { x: -1.96, z: 0.60, rx: 0.36, rz: 0.24, color: [0.16, 0.20, 0.24], alpha: 0.26, seed: 707 },
    MistPatch { x: -1.62, z: 0.46, rx: 0.28, rz: 0.22, color: [0.08, 0.66, 0.80], alpha: 0.22, seed: 808 },
    MistPatch { x: -1.46, z: 0.54, rx: 0.26, rz: 0.20, color: [0.88, 0.38, 0.16], alpha: 0.21, seed: 909 },
    MistPatch { x: -2.14, z: 0.66, rx: 0.26, rz: 0.20, color: [0.78, 0.64, 0.18], alpha: 0.18, seed: 997 },
];

fn sample_overspray_patina(wx: f32, wz: f32, base_rgb: [f32; 3]) -> ([f32; 3], f32) {
    let mut rgb = base_rgb;
    let mut paint_deposit = 0.0_f32;

    for patch in &STUDIO_OVERSPRAY_PATCHES {
        let dx = (wx - patch.x) / patch.rx;
        let dz = (wz - patch.z) / patch.rz;
        let r2 = dx * dx + dz * dz;
        if r2 < 1.35 {
            let radial = (1.0 - r2 / 1.35).clamp(0.0, 1.0);
            let breakup = fbm(wx * 9.5 + 1.7, wz * 9.5 - 2.3, patch.seed);
            let w = (patch.alpha * radial * radial * (0.48 + 0.90 * breakup)).clamp(0.0, 0.42);
            rgb[0] = rgb[0] * (1.0 - w) + patch.color[0] * w;
            rgb[1] = rgb[1] * (1.0 - w) + patch.color[1] * w;
            rgb[2] = rgb[2] * (1.0 - w) + patch.color[2] * w;
            paint_deposit += w;
        }
    }

    // Crisp micro paint droplets in active work zones (canvas drip apron & rack/cap-swap zone)
    let in_canvas_apron = wx.abs() < 1.38 && (0.06..=0.68).contains(&wz);
    let in_rack_zone = (-2.38..=-1.28).contains(&wx) && (0.24..=0.86).contains(&wz);
    if in_canvas_apron || in_rack_zone {
        let cell_scale = 18.0_f32;
        let gx = (wx * cell_scale).floor() as i32;
        let gz = (wz * cell_scale).floor() as i32;
        let fx = (wx * cell_scale).fract() - 0.5;
        let fz = (wz * cell_scale).fract() - 0.5;
        let jx = (hash2(gx, gz, 1301) - 0.5) * 0.55;
        let jz = (hash2(gx, gz, 1307) - 0.5) * 0.55;
        let spawn = hash2(gx, gz, 1319);
        if spawn > 0.85 {
            let dist = ((fx - jx).powi(2) + (fz - jz).powi(2)).sqrt();
            let rad = 0.11 + 0.13 * hash2(gx, gz, 1327);
            if dist < rad {
                let edge = ((rad - dist) / (rad * 0.35)).clamp(0.0, 1.0);
                let palette = [
                    [0.08, 0.60, 0.84],
                    [0.88, 0.36, 0.14],
                    [0.82, 0.22, 0.44],
                    [0.86, 0.66, 0.16],
                    [0.14, 0.15, 0.17],
                ];
                let idx = (hash2(gx, gz, 1361) * palette.len() as f32) as usize;
                let col = palette[idx.min(palette.len() - 1)];
                let alpha = 0.72 * edge;
                rgb[0] = rgb[0] * (1.0 - alpha) + col[0] * alpha;
                rgb[1] = rgb[1] * (1.0 - alpha) + col[1] * alpha;
                rgb[2] = rgb[2] * (1.0 - alpha) + col[2] * alpha;
                paint_deposit += alpha * 0.5;
            }
        }
    }

    (rgb, paint_deposit.clamp(0.0, 0.5))
}

fn build_concrete_floor_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            // World coordinates matching Plane3d(8.0, 6.5) at (0.0, 0.0, 3.21)
            let wx = (u - 0.5) * 8.0;
            let wz = 3.21 + (v - 0.5) * 6.5;

            let macro_n = fbm(u * 10.0, v * 8.0, 17);
            let micro_n = fbm(u * 48.0, v * 40.0, 93);
            let scuff = (fbm(u * 18.0 + 3.1, v * 6.0, 211) - 0.52).clamp(0.0, 1.0);

            // Concrete expansion slab joints every 25% of UV (2.0m x 1.625m slabs)
            let gx = ((u * 4.0).fract() - 0.5).abs();
            let gy = ((v * 4.0).fract() - 0.5).abs();
            let seam = if gx > 0.494 || gy > 0.494 { 1.0 } else { 0.0 };

            let base_luma = (0.32 + 0.11 * macro_n + 0.06 * micro_n - 0.08 * scuff - 0.11 * seam)
                .clamp(0.11, 0.58);

            let base_rgb = [base_luma * 0.99, base_luma * 1.00, base_luma * 1.03];
            let (rgb, deposit) = sample_overspray_patina(wx, wz, base_rgb);

            albedo.push((rgb[0] * 255.0).clamp(0.0, 255.0).round() as u8);
            albedo.push((rgb[1] * 255.0).clamp(0.0, 255.0).round() as u8);
            albedo.push((rgb[2] * 255.0).clamp(0.0, 255.0).round() as u8);
            albedo.push(255);

            // Roughness in Green channel (glTF/Bevy metallic-roughness convention: G = roughness, B = metallic)
            // Sealed studio floor: smooth epoxy sheen (~0.34..0.52) on intact concrete, matte in scuffs & seams
            let r_val = (0.40 + 0.24 * scuff + 0.14 * micro_n + 0.20 * seam - 0.12 * deposit)
                .clamp(0.32, 0.88);
            let r_u8 = (r_val * 255.0).round() as u8;
            rough.extend_from_slice(&[0, r_u8, 0, 255]);

            heights[(y * size + x) as usize] = macro_n * 0.4 + micro_n * 0.6 - seam * 0.8;
        }
    }

    let normal = normal_map_from_heights(&heights, size, 1.8);
    create_mipmapped_pbr_triplet(size, size, albedo, rough, normal, false, false)
}

fn build_drop_cloth_textures(width: u32, height: u32) -> (Image, Image, Image) {
    let n = (width * height) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            // World coordinates matching Plane3d(2.84, 0.68) at (0.0, 0.0026, 0.38)
            let wx = (u - 0.5) * 2.84;
            let wz = 0.38 + (v - 0.5) * 0.68;

            // Kraft-paper / heavy painter's canvas creases & weave
            let crease_x = (fbm(u * 9.0, v * 3.5, 401) - 0.5).abs();
            let crease_ridge = (1.0 - (crease_x * 9.0).clamp(0.0, 1.0)).powi(3);
            let fiber = fbm(u * 85.0, v * 28.0, 409);
            let weave = ((u * 260.0 * PI).sin() * (v * 70.0 * PI).sin()) * 0.5 + 0.5;

            let kraft_base = [
                (0.64 + 0.07 * fiber - 0.04 * crease_ridge).clamp(0.42, 0.80),
                (0.55 + 0.06 * fiber - 0.04 * crease_ridge).clamp(0.36, 0.72),
                (0.42 + 0.05 * fiber - 0.03 * crease_ridge).clamp(0.28, 0.58),
            ];

            let (mut rgb, deposit) = sample_overspray_patina(wx, wz, kraft_base);

            // Painter's blue masking tape border along perimeter edges (with subtle corner overlaps)
            let edge_u = u.min(1.0 - u) * 2.84;
            let edge_v = v.min(1.0 - v) * 0.68;
            let tape_w = 0.036_f32;
            let wobble = (fbm(u * 22.0, v * 12.0, 521) - 0.5) * 0.004;
            let on_tape = edge_u < (tape_w + wobble) || edge_v < (tape_w + wobble);

            let mut r_val = (0.84 + 0.08 * weave - 0.16 * deposit).clamp(0.42, 0.96);
            let mut h_val = fiber * 0.35 + crease_ridge * 0.55 + weave * 0.12;

            if on_tape {
                let tape_tex = fbm(u * 60.0, v * 24.0, 541);
                let tape_rgb = [
                    0.16 + 0.04 * tape_tex,
                    0.41 + 0.05 * tape_tex,
                    0.66 + 0.05 * tape_tex,
                ];
                // Subtle overspray can also sit on top of the masking tape
                rgb = [
                    tape_rgb[0] * 0.82 + rgb[0] * 0.18,
                    tape_rgb[1] * 0.82 + rgb[1] * 0.18,
                    tape_rgb[2] * 0.82 + rgb[2] * 0.18,
                ];
                r_val = 0.46 + 0.08 * tape_tex;
                h_val = 0.72 + 0.08 * tape_tex;
            }

            albedo.push((rgb[0] * 255.0).clamp(0.0, 255.0).round() as u8);
            albedo.push((rgb[1] * 255.0).clamp(0.0, 255.0).round() as u8);
            albedo.push((rgb[2] * 255.0).clamp(0.0, 255.0).round() as u8);
            albedo.push(255);

            let r_u8 = (r_val * 255.0).round() as u8;
            rough.extend_from_slice(&[0, r_u8, 0, 255]);
            heights[y as usize * width as usize + x as usize] = h_val;
        }
    }

    // Rectangular normal map helper
    let mut normal = Vec::with_capacity(n * 4);
    let w_i = width as i32;
    let h_i = height as i32;
    for y in 0..h_i {
        for x in 0..w_i {
            let xm = (x - 1).clamp(0, w_i - 1) as usize;
            let xp = (x + 1).clamp(0, w_i - 1) as usize;
            let ym = (y - 1).clamp(0, h_i - 1) as usize;
            let yp = (y + 1).clamp(0, h_i - 1) as usize;
            let row = y as usize * width as usize;
            let dx = heights[row + xp] - heights[row + xm];
            let dy = heights[yp * width as usize + x as usize]
                - heights[ym * width as usize + x as usize];
            let n_vec = Vec3::new(-dx * 1.6, -dy * 1.6, 1.0).normalize();
            normal.push(((n_vec.x * 0.5 + 0.5) * 255.0).round() as u8);
            normal.push(((n_vec.y * 0.5 + 0.5) * 255.0).round() as u8);
            normal.push(((n_vec.z * 0.5 + 0.5) * 255.0).round() as u8);
            normal.push(255);
        }
    }

    create_mipmapped_pbr_triplet(width, height, albedo, rough, normal, false, false)
}

fn build_corrugated_panel_textures(size: u32, ribs_along_v: bool) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    let rib_count = 22.0_f32;
    let panel_seams = 4.0_f32;

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let (across, along) = if ribs_along_v { (v, u) } else { (u, v) };

            // Trapezoidal corrugated industrial shop wall profile
            let phase = (across * rib_count).fract();
            let rib_h = if phase < 0.25 {
                phase / 0.25
            } else if phase < 0.55 {
                1.0
            } else if phase < 0.80 {
                1.0 - (phase - 0.55) / 0.25
            } else {
                0.0
            };

            let seam_across = ((across * panel_seams).fract() - 0.5).abs() > 0.492;
            let seam_along = ((along * 2.0).fract() - 0.5).abs() > 0.494;
            let seam = if seam_across || seam_along { 1.0 } else { 0.0 };

            let grime = fbm(u * 14.0, v * 14.0, 613);
            let micro = fbm(u * 48.0, v * 48.0, 619);

            // Subtle lower-wall shop dust gradient near the floor
            let height_norm = if ribs_along_v { u } else { 1.0 - v };
            let floor_dust = (1.0 - height_norm * 3.0).clamp(0.0, 1.0) * 0.04;

            let luma = (0.21 + 0.035 * rib_h + 0.035 * grime + floor_dust - 0.06 * seam)
                .clamp(0.10, 0.34);

            albedo.push((luma * 0.96 * 255.0).round() as u8);
            albedo.push((luma * 1.00 * 255.0).round() as u8);
            albedo.push((luma * 1.05 * 255.0).round() as u8);
            albedo.push(255);

            let r_val = (0.64 + 0.16 * grime + 0.08 * micro + 0.12 * seam - 0.06 * rib_h)
                .clamp(0.45, 0.92);
            let m_u8 = if seam > 0.5 { 20 } else { 48 };
            rough.extend_from_slice(&[0, (r_val * 255.0).round() as u8, m_u8, 255]);

            heights[(y * size + x) as usize] = rib_h * 0.85 + micro * 0.15 - seam * 0.6;
        }
    }

    let normal = normal_map_from_heights(&heights, size, 2.4);
    create_mipmapped_pbr_triplet(size, size, albedo, rough, normal, false, false)
}

fn build_ceiling_deck_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            // Ribbed roof deck flutes along X (spanning Z)
            let phase = (v * 24.0).fract();
            let flute = ((phase * 2.0 * PI).sin() * 0.5 + 0.5).powf(1.4);
            let bay_seam = ((u * 4.0).fract() - 0.5).abs() > 0.492;
            let noise = fbm(u * 18.0, v * 18.0, 733);

            let luma = (0.12 + 0.03 * flute + 0.025 * noise - if bay_seam { 0.04 } else { 0.0 })
                .clamp(0.06, 0.22);

            albedo.push((luma * 0.95 * 255.0).round() as u8);
            albedo.push((luma * 0.99 * 255.0).round() as u8);
            albedo.push((luma * 1.06 * 255.0).round() as u8);
            albedo.push(255);

            let r_val = (0.68 + 0.16 * noise - 0.08 * flute).clamp(0.48, 0.92);
            rough.extend_from_slice(&[0, (r_val * 255.0).round() as u8, 72, 255]);

            heights[(y * size + x) as usize] = flute * 0.8 + noise * 0.2;
        }
    }

    let normal = normal_map_from_heights(&heights, size, 2.0);
    create_mipmapped_pbr_triplet(size, size, albedo, rough, normal, false, false)
}

fn build_industrial_wall_textures(width: u32, height: u32) -> (Image, Image, Image) {
    let n = (width * height) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    let cols = 16.0_f32;
    let rows = 21.0_f32;

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;
            // World coordinates spanning back wall: X in [-4.0, +4.0], Y in [0.0, 4.2]
            let wx = (u - 0.5) * 8.0;
            let wy = (1.0 - v) * 4.2;

            // Organic mortar line warp + chipped CMU block imperfections
            let warp_u = (fbm(wx * 3.8, wy * 3.8, 311) - 0.5) * 0.045;
            let warp_v = (fbm(wx * 4.2 + 5.1, wy * 4.2, 317) - 0.5) * 0.055;
            let cu = u * cols + warp_u;
            let rv = v * rows + warp_v;

            let row_idx = rv.floor();
            let row_f = rv.fract();
            let stagger = if (row_idx as i32).rem_euclid(2) == 0 { 0.0 } else { 0.5 };
            let col_idx = (cu + stagger).floor();
            let col_f = (cu + stagger).fract();

            let chip_n = fbm(wx * 14.0, wy * 14.0, 331);
            let mortar_thick_v = 0.068 + 0.028 * (chip_n - 0.5);
            let mortar_thick_u = 0.032 + 0.018 * (chip_n - 0.5);
            let mortar = row_f < mortar_thick_v
                || row_f > (1.0 - mortar_thick_v)
                || col_f < mortar_thick_u
                || col_f > (1.0 - mortar_thick_u);

            let block_id = (col_idx as i32).wrapping_mul(37) + (row_idx as i32).wrapping_mul(19);
            let block_tint = hash2(block_id, 5, 777) * 0.065;
            let aggregate = fbm(wx * 18.0, wy * 18.0, 349);
            let micro_pit = fbm(wx * 52.0, wy * 52.0, 353);
            let paint_patch = fbm(wx * 0.75 + 2.3, wy * 0.65 - 1.1, 367);

            // Lower-wall concrete floor kickup dust & faint vertical ceiling moisture streaks
            let low_grime = (1.0 - wy / 0.85).clamp(0.0, 1.0).powf(1.6) * (0.045 + 0.035 * aggregate);
            let streak = (fbm(wx * 2.4, wy * 0.35, 383) - 0.56).clamp(0.0, 1.0)
                * (1.0 - wy / 4.2).clamp(0.2, 1.0)
                * 0.14;

            let base_luma = if mortar {
                0.21 + 0.045 * micro_pit - 0.025 * low_grime
            } else {
                0.275 + block_tint + 0.06 * aggregate + 0.04 * paint_patch - low_grime - streak
            };

            let warm_cool = 0.012 * (hash2(block_id, 11, 797) - 0.5);
            let mut rgb = [
                (base_luma * (1.01 + warm_cool)).clamp(0.11, 0.50),
                (base_luma * 1.00).clamp(0.11, 0.50),
                (base_luma * (1.03 - warm_cool)).clamp(0.11, 0.50),
            ];

            // Subtle dried studio overspray halo drifting just outside canvas frame perimeter
            let dx_edge = (wx.abs() - 1.25).clamp(0.0, 0.45);
            let outside_y = if wy < 0.80 {
                0.80 - wy
            } else if wy > 2.30 {
                wy - 2.30
            } else {
                0.0
            };
            let rim_dist = (dx_edge * dx_edge + outside_y * outside_y).sqrt();
            let halo_band = if (wx.abs() > 1.20 || wy < 0.84 || wy > 2.26) && rim_dist < 0.34 {
                ((1.0 - rim_dist / 0.34).clamp(0.0, 1.0)).powf(2.0)
            } else {
                0.0
            };
            let mut mist_w = 0.0_f32;
            if halo_band > 0.02 {
                let mist_mix = fbm(wx * 4.5, wy * 4.5, 409);
                mist_w = (halo_band * mist_mix * 0.42).clamp(0.0, 0.24);
                let ghost_col = if wx < 0.0 {
                    [0.14, 0.52, 0.68]
                } else {
                    [0.72, 0.30, 0.26]
                };
                for c in 0..3 {
                    rgb[c] = rgb[c] * (1.0 - mist_w) + ghost_col[c] * mist_w;
                }
            }

            albedo.push((rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8);
            albedo.push((rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8);
            albedo.push((rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8);
            albedo.push(255);

            // Chalky matte mortar (0.93) vs subtly satin roller-painted CMU blocks (0.68..0.85)
            let rough = if mortar {
                (0.92 + 0.04 * micro_pit).clamp(0.86, 0.97)
            } else {
                (0.70 + 0.14 * (1.0 - paint_patch) + 0.08 * micro_pit + 0.08 * mist_w)
                    .clamp(0.64, 0.92)
            };
            rough_metal.extend_from_slice(&[0, (rough * 255.0).round() as u8, 4, 255]);

            heights[(y * width + x) as usize] = if mortar {
                -0.55 + micro_pit * 0.15
            } else {
                aggregate * 0.35 + micro_pit * 0.35 - chip_n * 0.10
            };
        }
    }

    let normal = normal_map_from_heights_rect(&heights, width, height, 2.4);
    create_mipmapped_pbr_triplet(width, height, albedo, rough_metal, normal, false, false)
}

fn build_wood_grain_textures(size: u32) -> (Image, Image, Image) {
    let n = (size * size) as usize;
    let mut albedo = Vec::with_capacity(n * 4);
    let mut rough_metal = Vec::with_capacity(n * 4);
    let mut heights = vec![0.0_f32; n];

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;
            let warp = fbm(u * 4.5, v * 1.4, 811) * 0.25;
            let ring_coord = (v * 14.0 + warp * 12.0 + fbm(u * 12.0, v * 3.0, 823) * 2.5) * PI;
            let ring = ring_coord.sin() * 0.5 + 0.5;
            let fiber = value_noise(u * 120.0, v * 28.0, 829);

            let knot_d = (((u - 0.62) * 3.2).powi(2) + ((v - 0.42) * 8.0).powi(2)).sqrt();
            let knot = (1.0 - (knot_d / 0.28).clamp(0.0, 1.0)).powi(2);

            let luma = (0.58 + 0.12 * ring + 0.05 * fiber - 0.22 * knot).clamp(0.24, 0.80);
            albedo.extend_from_slice(&[
                (luma * 1.12 * 255.0).round().clamp(0.0, 255.0) as u8,
                (luma * 0.86 * 255.0).round().clamp(0.0, 255.0) as u8,
                (luma * 0.56 * 255.0).round().clamp(0.0, 255.0) as u8,
                255,
            ]);

            let rough = (0.68 + 0.14 * (1.0 - ring) + 0.08 * knot).clamp(0.55, 0.90);
            rough_metal.extend_from_slice(&[0, (rough * 255.0).round() as u8, 0, 255]);
            heights[(y * size + x) as usize] = ring * 0.55 + fiber * 0.35 - knot * 0.45;
        }
    }

    let normal = normal_map_from_heights(&heights, size, 1.5);
    create_mipmapped_pbr_triplet(size, size, albedo, rough_metal, normal, false, false)
}

fn build_light_shaft_texture(width: u32, height: u32) -> Image {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let v = y as f32 / (height.saturating_sub(1).max(1)) as f32;
        // Soft entrance near lens (v -> 0) and smooth feather to 0 well before beam end (v -> 1)
        let axial = (v / 0.14).clamp(0.0, 1.0) * (1.0 - v).powf(2.2);
        for x in 0..width {
            let u = x as f32 / width as f32;
            let streaks = 0.86 + 0.14 * (u * 10.0 * PI).sin();
            let alpha = (axial * streaks).clamp(0.0, 1.0);
            data.extend_from_slice(&[255, 255, 255, (alpha * 255.0).round() as u8]);
        }
    }
    create_mipmapped_rgba8_image(
        width,
        height,
        data,
        TextureFormat::Rgba8UnormSrgb,
        MipmapKind::SrgbColor,
        false,
        false,
    )
}

/// Builds an open-ended frustum mesh (`y = +height/2` at `r_top`, `y = -height/2` at `r_bot`)
/// with UVs `(u, v = 0 at top -> 1 at bottom)` for feathered volumetric spotlight cones.
fn make_open_cone_mesh(r_top: f32, r_bot: f32, height: f32, segments: usize) -> Mesh {
    let segs = segments.max(8);
    let mut positions = Vec::with_capacity((segs + 1) * 2);
    let mut normals = Vec::with_capacity((segs + 1) * 2);
    let mut uvs = Vec::with_capacity((segs + 1) * 2);
    let mut indices = Vec::with_capacity(segs * 6);

    let y_top = height * 0.5;
    let y_bot = -height * 0.5;
    let slope = (r_bot - r_top) / height.max(1e-4);

    for i in 0..=segs {
        let u = i as f32 / segs as f32;
        let theta = u * 2.0 * PI;
        let (st, ct) = theta.sin_cos();
        let n = Vec3::new(ct, slope, st).normalize();

        positions.push([r_top * ct, y_top, r_top * st]);
        normals.push([n.x, n.y, n.z]);
        uvs.push([u, 0.0]);

        positions.push([r_bot * ct, y_bot, r_bot * st]);
        normals.push([n.x, n.y, n.z]);
        uvs.push([u, 1.0]);
    }

    for i in 0..segs as u32 {
        let a = i * 2;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;
        indices.extend_from_slice(&[a, b, c, c, b, d]);
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

fn build_static_hose_tube_mesh(
    p0: Vec3,
    m0: Vec3,
    p1: Vec3,
    m1: Vec3,
    radius: f32,
    rings: usize,
) -> Mesh {
    const SIDES: usize = 10;
    let rings = rings.max(4);
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
        pt.y = pt.y.max(radius + 0.004);
        pts.push(pt);
    }

    let mut positions = Vec::with_capacity(rings * (SIDES + 1));
    let mut normals = Vec::with_capacity(rings * (SIDES + 1));
    let mut uvs = Vec::with_capacity(rings * (SIDES + 1));
    let mut indices = Vec::with_capacity((rings - 1) * SIDES * 6);

    let mut tangents = Vec::with_capacity(rings);
    for i in 0..rings {
        let t = if i == 0 {
            (pts[1] - pts[0]).normalize_or(Vec3::Y)
        } else if i + 1 == rings {
            (pts[i] - pts[i - 1]).normalize_or(Vec3::Y)
        } else {
            (pts[i + 1] - pts[i - 1]).normalize_or(Vec3::Y)
        };
        tangents.push(t);
    }

    let t0 = tangents[0];
    let hint = if t0.dot(Vec3::X).abs() < 0.85 {
        Vec3::X
    } else {
        Vec3::Z
    };
    let mut u_ax = (hint - t0 * hint.dot(t0)).normalize_or(Vec3::Z);

    for i in 0..rings {
        let ti = tangents[i];
        if i > 0 {
            let t_prev = tangents[i - 1];
            if t_prev.dot(ti) > -0.9999 {
                u_ax = Quat::from_rotation_arc(t_prev, ti) * u_ax;
            }
            u_ax = (u_ax - ti * u_ax.dot(ti)).normalize_or(Vec3::X);
        }
        let v_ax = ti.cross(u_ax).normalize_or(Vec3::Z);
        let v = i as f32 / (rings - 1) as f32;

        for j in 0..=SIDES {
            let u = j as f32 / SIDES as f32;
            let (st, ct) = (u * 2.0 * PI).sin_cos();
            let n = (u_ax * ct + v_ax * st).normalize();
            let p = pts[i] + n * radius;
            positions.push([p.x, p.y, p.z]);
            normals.push([n.x, n.y, n.z]);
            uvs.push([u, v]);
        }
    }

    let stride = (SIDES + 1) as u32;
    for i in 0..(rings - 1) as u32 {
        let r0 = i * stride;
        let r1 = (i + 1) * stride;
        for j in 0..SIDES as u32 {
            let a = r0 + j;
            let b = r1 + j;
            let c = r0 + j + 1;
            let d = r1 + j + 1;
            indices.extend_from_slice(&[a, c, b, c, d, b]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

fn normal_map_from_heights(heights: &[f32], size: u32, strength: f32) -> Vec<u8> {
    normal_map_from_heights_rect(heights, size, size, strength)
}

fn normal_map_from_heights_rect(
    heights: &[f32],
    width: u32,
    height: u32,
    strength: f32,
) -> Vec<u8> {
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

/// Builds a 6-face `Rgba16Float` HDR studio warehouse cubemap with skylight softboxes,
/// warm industrial wall bounce, and dark concrete floor reflections.
fn build_studio_hdr_cubemap(face_size: u32) -> Image {
    let mut data = Vec::with_capacity((face_size * face_size * 6 * 8) as usize);

    for face in 0..6 {
        for y in 0..face_size {
            let v = 2.0 * ((y as f32 + 0.5) / face_size as f32) - 1.0;
            for x in 0..face_size {
                let u = 2.0 * ((x as f32 + 0.5) / face_size as f32) - 1.0;
                let dir = cubemap_face_dir(face, u, v).normalize();
                let hdr = sample_studio_hdr(dir);
                for ch in hdr {
                    let bits = f32_to_f16_bits(ch);
                    data.extend_from_slice(&bits.to_le_bytes());
                }
            }
        }
    }

    Image {
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        }),
        ..Image::new(
            Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba16Float,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}

#[inline]
fn cubemap_face_dir(face: usize, u: f32, v: f32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),  // +X
        1 => Vec3::new(-1.0, -v, u),  // -X
        2 => Vec3::new(u, 1.0, v),    // +Y
        3 => Vec3::new(u, -1.0, -v),  // -Y
        4 => Vec3::new(u, -v, 1.0),   // +Z
        _ => Vec3::new(-u, -v, -1.0), // -Z
    }
}

fn sample_studio_hdr(dir: Vec3) -> [f32; 4] {
    // High-contrast studio gradient: dark concrete floor bounce -> warm/cool side walls -> dark ceiling deck
    let mut rgb = if dir.y < 0.0 {
        let f = (-dir.y).clamp(0.0, 1.0);
        Vec3::new(0.14, 0.14, 0.15).lerp(Vec3::new(0.035, 0.038, 0.045), f.sqrt())
    } else {
        let t_up = dir.y.clamp(0.0, 1.0);
        // Subtle warm right-wall (+X) vs cooler left-wall (-X) industrial bounce
        let warm_mix = (dir.x * 0.5 + 0.5).clamp(0.0, 1.0);
        let horizon = Vec3::new(0.15, 0.19, 0.25).lerp(Vec3::new(0.28, 0.21, 0.15), warm_mix);
        horizon.lerp(Vec3::new(0.09, 0.105, 0.135), t_up.powf(0.75))
    };

    // Distinct overhead bright cool-white fluorescent/LED shop light strips for crisp metallic reflections
    if dir.y > 0.24 {
        let px = dir.x / dir.y;
        let pz = dir.z / dir.y;
        let strip_centers_x = [-0.52_f32, 0.52];
        let strip_centers_z = [-0.65_f32, 0.10, 0.85];
        for &sx in &strip_centers_x {
            for &sz in &strip_centers_z {
                let dx = ((px - sx).abs() / 0.28).clamp(0.0, 1.5);
                let dz = ((pz - sz).abs() / 0.055).clamp(0.0, 1.5);
                let mx = 1.0 - ((dx - 0.80) / 0.25).clamp(0.0, 1.0);
                let mz = 1.0 - ((dz - 0.72) / 0.35).clamp(0.0, 1.0);
                let strip = mx * mz;
                if strip > 0.0 {
                    rgb += Vec3::new(10.5, 11.2, 12.6) * strip;
                }
            }
        }

        // Central diffused skylight softbox fill strip
        let cx = 1.0 - ((px.abs() - 0.26) / 0.14).clamp(0.0, 1.0);
        let cz = 1.0 - (((pz + 0.15).abs() - 0.42) / 0.16).clamp(0.0, 1.0);
        rgb += Vec3::new(3.2, 3.5, 4.0) * (cx * cz);
    }

    // Warm tungsten/amber side work-light bounce & rim highlight on +X / +Z side
    if dir.x > 0.35 && dir.y > -0.05 && dir.y < 0.55 {
        let x_w = ((dir.x - 0.35) / 0.55).clamp(0.0, 1.0);
        let y_w = (1.0 - (dir.y - 0.24).abs() / 0.30).clamp(0.0, 1.0);
        rgb += Vec3::new(2.8, 1.75, 0.85) * (x_w * y_w);
    }

    // Cooler slate/steel side bounce on -X / -Z side
    if dir.x < -0.45 && dir.y > -0.05 && dir.y < 0.50 {
        let x_w = ((-dir.x - 0.45) / 0.50).clamp(0.0, 1.0);
        let y_w = (1.0 - (dir.y - 0.22).abs() / 0.28).clamp(0.0, 1.0);
        rgb += Vec3::new(0.45, 0.65, 0.95) * (x_w * y_w);
    }

    let _ = PI;
    [rgb.x, rgb.y, rgb.z, 1.0]
}

/// Converts an `f32` into IEEE 754 binary16 (`f16`) bit representation.
#[inline]
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN
        return sign | 0x7c00 | if frac != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (frac | 0x0080_0000) >> (14 - new_exp);
        return sign | (m as u16);
    }

    sign | ((new_exp as u16) << 10) | ((frac >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_hdr_cubemap_is_valid_6_face_cube() {
        let img = build_studio_hdr_cubemap(64);
        assert_eq!(img.texture_descriptor.size.width, 64);
        assert_eq!(img.texture_descriptor.size.height, 64);
        assert_eq!(img.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(img.texture_descriptor.dimension, TextureDimension::D2);
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rgba16Float);
        assert_eq!(
            img.texture_view_descriptor.as_ref().and_then(|d| d.dimension),
            Some(TextureViewDimension::Cube)
        );
        assert_eq!(img.data.as_ref().map(|d| d.len()), Some(64 * 64 * 6 * 8));
    }

    #[test]
    fn f16_conversion_handles_zero_one_and_hdr() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(8.0), 0x4800);
    }

    #[test]
    fn cap_station_collet_cups_align_below_nozzle_tip_and_clear_rack() {
        let studio = StudioConfig::default();
        let docks = studio.cap_station_poses(16);
        let rack_origin = Vec3::new(studio.rack_center_xz.x, 0.0, studio.rack_center_xz.z);
        let rack_rot_inv = Quat::from_rotation_y(studio.rack_yaw_deg.to_radians()).inverse();
        let rack_half_w = (8.0 * studio.rack_slot_spacing + 0.070) * 0.5;

        for dock in &docks {
            let nozzle_world = cap_dock_nozzle_world(dock);
            let nozzle_local = rack_rot_inv * (nozzle_world - rack_origin);
            let y_cup = nozzle_local.y - 0.004;
            assert!((y_cup - (nozzle_world.y - 0.004)).abs() < 1e-5);
            // Manifold front fascia sits at z = 0.018, can body at dock sits at z >= 0.060 - 0.033 = 0.027
            assert!(nozzle_local.z - 0.027 > 0.010);
            assert!(nozzle_local.x > rack_half_w + 0.05);
        }
    }

    #[test]
    fn studio_surface_procedural_textures_have_expected_dimensions_and_patina() {
        let (dc_a, dc_r, dc_n) = build_drop_cloth_textures(128, 64);
        assert_eq!(dc_a.texture_descriptor.size.width, 128);
        assert_eq!(dc_a.texture_descriptor.size.height, 64);
        assert_eq!(dc_r.texture_descriptor.size.width, 128);
        assert_eq!(dc_n.texture_descriptor.size.height, 64);

        let (wall_a, wall_r, wall_n) = build_corrugated_panel_textures(64, true);
        assert_eq!(wall_a.texture_descriptor.size.width, 64);
        assert_eq!(wall_r.texture_descriptor.size.height, 64);
        assert_eq!(wall_n.texture_descriptor.size.width, 64);

        let (ceil_a, ceil_r, ceil_n) = build_ceiling_deck_textures(64);
        assert_eq!(ceil_a.texture_descriptor.size.width, 64);
        assert_eq!(ceil_r.texture_descriptor.size.height, 64);
        assert_eq!(ceil_n.texture_descriptor.size.width, 64);

        let (wall_a, wall_rm, wall_wn) = build_industrial_wall_textures(128, 64);
        assert_eq!(wall_a.texture_descriptor.size.width, 128);
        assert_eq!(wall_rm.texture_descriptor.size.height, 64);
        assert_eq!(wall_wn.texture_descriptor.size.width, 128);

        let (wood_a, wood_r, wood_n) = build_wood_grain_textures(64);
        assert_eq!(wood_a.texture_descriptor.size.width, 64);
        assert_eq!(wood_r.texture_descriptor.size.height, 64);
        assert_eq!(wood_n.texture_descriptor.size.width, 64);

        let shaft = build_light_shaft_texture(32, 64);
        assert_eq!(shaft.texture_descriptor.size.width, 32);
        assert_eq!(shaft.texture_descriptor.size.height, 64);

        let dc_mesh = build_wrinkled_drop_cloth_mesh(2.84, 0.68, 24, 8);
        let pos_attr = dc_mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .expect("drop cloth positions");
        if let bevy::mesh::VertexAttributeValues::Float32x3(pts) = pos_attr {
            // Perimeter border vertices stay flush at y == 0 while interior wrinkles gently
            assert!(pts[0][1].abs() < 1e-6);
            assert!(pts.iter().any(|p| p[1] > 0.0005 && p[1] < 0.01));
        } else {
            panic!("expected Float32x3 positions");
        }

        // Overspray patina deposits subtly under canvas & rack without altering far floor
        let (_, dep_canvas) = sample_overspray_patina(0.18, 0.14, [0.3, 0.3, 0.3]);
        let (_, dep_rack) = sample_overspray_patina(-1.96, 0.60, [0.3, 0.3, 0.3]);
        let (_, dep_far) = sample_overspray_patina(3.2, 4.5, [0.3, 0.3, 0.3]);
        assert!(dep_canvas > 0.02);
        assert!(dep_rack > 0.02);
        assert_eq!(dep_far, 0.0);
    }
}

