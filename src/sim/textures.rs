use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::Vec3,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

/// Filtering mode used when constructing a full CPU mipmap chain for a 2D RGBA8 texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipmapKind {
    SrgbColor,
    LinearData,
    NormalMap,
}

/// Returns the full mip-level count down to `1x1` for a `width x height` 2D texture,
/// matching `wgpu::Extent3d::max_mips(TextureDimension::D2)`.
#[inline]
pub fn mip_level_count_2d(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// Returns the total byte length of a `LayerMajor` RGBA8 mip chain from mip 0 to `1x1`.
pub fn mip_chain_byte_len_rgba8(width: u32, height: u32) -> usize {
    let levels = mip_level_count_2d(width.max(1), height.max(1));
    let mut total = 0usize;
    for level in 0..levels {
        let w = (width >> level).max(1) as usize;
        let h = (height >> level).max(1) as usize;
        total += w * h * 4;
    }
    total
}

/// Creates a `RenderAssetUsages::RENDER_WORLD` 2D RGBA8 Bevy [`Image`] containing a complete
/// high-quality CPU mipmap chain down to `1x1` with trilinear + 8x anisotropic filtering.
pub fn create_mipmapped_rgba8_image(
    width: u32,
    height: u32,
    level0_data: Vec<u8>,
    format: TextureFormat,
    kind: MipmapKind,
    repeat_u: bool,
    repeat_v: bool,
) -> bevy::image::Image {
    assert!(width > 0 && height > 0, "texture dimensions must be > 0");
    assert_eq!(
        level0_data.len(),
        width as usize * height as usize * 4,
        "level-0 RGBA8 buffer size mismatch"
    );

    let levels = mip_level_count_2d(width, height);
    let mut chain = Vec::with_capacity(mip_chain_byte_len_rgba8(width, height));
    chain.extend_from_slice(&level0_data);

    let mut src = level0_data;
    let mut src_w = width;
    let mut src_h = height;

    for _ in 1..levels {
        let dst_w = (src_w >> 1).max(1);
        let dst_h = (src_h >> 1).max(1);
        let next = match kind {
            MipmapKind::SrgbColor => downsample_srgb_2x2(&src, src_w, src_h, dst_w, dst_h),
            MipmapKind::LinearData => downsample_linear_2x2(&src, src_w, src_h, dst_w, dst_h),
            MipmapKind::NormalMap => downsample_normal_2x2(&src, src_w, src_h, dst_w, dst_h).0,
        };
        chain.extend_from_slice(&next);
        src = next;
        src_w = dst_w;
        src_h = dst_h;
    }

    assemble_mipmapped_image(width, height, format, levels, chain, repeat_u, repeat_v)
}

/// Creates a coordinated PBR texture triplet `(albedo_srgb, rough_metal, normal)` with full
/// mip chains and mild Toksvig normal-variance roughness widening at coarser mip levels so
/// minified metallic/clearcoated surfaces filter smoothly instead of sparkling.
pub fn create_mipmapped_pbr_triplet(
    width: u32,
    height: u32,
    albedo_srgb: Vec<u8>,
    rough_metal: Vec<u8>,
    normal: Vec<u8>,
    repeat_u: bool,
    repeat_v: bool,
) -> (
    bevy::image::Image,
    bevy::image::Image,
    bevy::image::Image,
) {
    assert!(width > 0 && height > 0, "texture dimensions must be > 0");
    let expected = width as usize * height as usize * 4;
    assert_eq!(albedo_srgb.len(), expected);
    assert_eq!(rough_metal.len(), expected);
    assert_eq!(normal.len(), expected);

    let levels = mip_level_count_2d(width, height);
    let total_bytes = mip_chain_byte_len_rgba8(width, height);

    let albedo_img = create_mipmapped_rgba8_image(
        width,
        height,
        albedo_srgb,
        TextureFormat::Rgba8UnormSrgb,
        MipmapKind::SrgbColor,
        repeat_u,
        repeat_v,
    );

    let mut rm_chain = Vec::with_capacity(total_bytes);
    let mut n_chain = Vec::with_capacity(total_bytes);
    rm_chain.extend_from_slice(&rough_metal);
    n_chain.extend_from_slice(&normal);

    let mut src_rm = rough_metal;
    let mut src_n = normal;
    let mut src_w = width;
    let mut src_h = height;

    for _ in 1..levels {
        let dst_w = (src_w >> 1).max(1);
        let dst_h = (src_h >> 1).max(1);
        let (next_n, avg_lengths) = downsample_normal_2x2(&src_n, src_w, src_h, dst_w, dst_h);
        let mut next_rm = downsample_linear_2x2(&src_rm, src_w, src_h, dst_w, dst_h);

        // Apply Toksvig roughness widening to G channel based on normal averaging contraction.
        for (i, &avg_len) in avg_lengths.iter().enumerate() {
            let variance_term = 0.35 * (1.0 - avg_len).max(0.0);
            let g_idx = i * 4 + 1;
            let r = next_rm[g_idx] as f32 / 255.0;
            let r_eff = (r * r + variance_term).sqrt().clamp(r, 0.95);
            next_rm[g_idx] = (r_eff * 255.0).round().clamp(0.0, 255.0) as u8;
        }

        rm_chain.extend_from_slice(&next_rm);
        n_chain.extend_from_slice(&next_n);
        src_rm = next_rm;
        src_n = next_n;
        src_w = dst_w;
        src_h = dst_h;
    }

    let rm_img = assemble_mipmapped_image(
        width,
        height,
        TextureFormat::Rgba8Unorm,
        levels,
        rm_chain,
        repeat_u,
        repeat_v,
    );
    let n_img = assemble_mipmapped_image(
        width,
        height,
        TextureFormat::Rgba8Unorm,
        levels,
        n_chain,
        repeat_u,
        repeat_v,
    );

    (albedo_img, rm_img, n_img)
}

fn assemble_mipmapped_image(
    width: u32,
    height: u32,
    format: TextureFormat,
    mip_level_count: u32,
    chain: Vec<u8>,
    repeat_u: bool,
    repeat_v: bool,
) -> bevy::image::Image {
    let mut img = bevy::image::Image::new_uninit(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.mip_level_count = mip_level_count;
    img.data = Some(chain);

    let mut desc = ImageSamplerDescriptor::linear();
    desc.set_anisotropic_filter(8);
    desc.address_mode_u = if repeat_u {
        ImageAddressMode::Repeat
    } else {
        ImageAddressMode::ClampToEdge
    };
    desc.address_mode_v = if repeat_v {
        ImageAddressMode::Repeat
    } else {
        ImageAddressMode::ClampToEdge
    };
    img.sampler = ImageSampler::Descriptor(desc);
    img
}

#[inline]
fn srgb_u8_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

#[inline]
fn sample_offsets_2x2(src_w: u32, src_h: u32, dst_x: u32, dst_y: u32) -> [usize; 4] {
    let sx0 = (dst_x * 2).min(src_w - 1) as usize;
    let sx1 = (dst_x * 2 + 1).min(src_w - 1) as usize;
    let sy0 = (dst_y * 2).min(src_h - 1) as usize;
    let sy1 = (dst_y * 2 + 1).min(src_h - 1) as usize;
    let stride = src_w as usize;
    [
        (sy0 * stride + sx0) * 4,
        (sy0 * stride + sx1) * 4,
        (sy1 * stride + sx0) * 4,
        (sy1 * stride + sx1) * 4,
    ]
}

fn downsample_srgb_2x2(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = Vec::with_capacity(dst_w as usize * dst_h as usize * 4);
    for y in 0..dst_h {
        for x in 0..dst_w {
            let offs = sample_offsets_2x2(src_w, src_h, x, y);
            let mut r = 0.0_f32;
            let mut g = 0.0_f32;
            let mut b = 0.0_f32;
            let mut a = 0_u32;
            for o in offs {
                r += srgb_u8_to_linear(src[o]);
                g += srgb_u8_to_linear(src[o + 1]);
                b += srgb_u8_to_linear(src[o + 2]);
                a += src[o + 3] as u32;
            }
            dst.push(linear_to_srgb_u8(r * 0.25));
            dst.push(linear_to_srgb_u8(g * 0.25));
            dst.push(linear_to_srgb_u8(b * 0.25));
            dst.push(((a + 2) >> 2) as u8);
        }
    }
    dst
}

fn downsample_linear_2x2(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = Vec::with_capacity(dst_w as usize * dst_h as usize * 4);
    for y in 0..dst_h {
        for x in 0..dst_w {
            let offs = sample_offsets_2x2(src_w, src_h, x, y);
            let mut sums = [0_u32; 4];
            for o in offs {
                sums[0] += src[o] as u32;
                sums[1] += src[o + 1] as u32;
                sums[2] += src[o + 2] as u32;
                sums[3] += src[o + 3] as u32;
            }
            for s in sums {
                dst.push(((s + 2) >> 2) as u8);
            }
        }
    }
    dst
}

fn downsample_normal_2x2(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> (Vec<u8>, Vec<f32>) {
    let texel_count = dst_w as usize * dst_h as usize;
    let mut dst = Vec::with_capacity(texel_count * 4);
    let mut lengths = Vec::with_capacity(texel_count);

    for y in 0..dst_h {
        for x in 0..dst_w {
            let offs = sample_offsets_2x2(src_w, src_h, x, y);
            let mut sum = Vec3::ZERO;
            for o in offs {
                let nx = src[o] as f32 / 127.5 - 1.0;
                let ny = src[o + 1] as f32 / 127.5 - 1.0;
                let nz = src[o + 2] as f32 / 127.5 - 1.0;
                sum += Vec3::new(nx, ny, nz);
            }
            let avg = sum * 0.25;
            let avg_len = avg.length().clamp(0.0, 1.0);
            lengths.push(avg_len);

            let unit = avg.normalize_or(Vec3::Z);
            dst.push(((unit.x * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8);
            dst.push(((unit.y * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8);
            dst.push(((unit.z * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8);
            dst.push(255);
        }
    }
    (dst, lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_counts_and_byte_lengths_match_wgpu_expectations() {
        assert_eq!(mip_level_count_2d(1, 1), 1);
        assert_eq!(mip_level_count_2d(64, 64), 7);
        assert_eq!(mip_level_count_2d(512, 64), 10);
        assert_eq!(mip_level_count_2d(512, 192), 10);
        assert_eq!(mip_level_count_2d(1280, 640), 11);

        // 4x4 -> 4x4 + 2x2 + 1x1 = (16 + 4 + 1) * 4 = 84 bytes
        assert_eq!(mip_chain_byte_len_rgba8(4, 4), 84);
        // 8x4 -> 8x4 + 4x2 + 2x1 + 1x1 = (32 + 8 + 2 + 1) * 4 = 172 bytes
        assert_eq!(mip_chain_byte_len_rgba8(8, 4), 172);
    }

    #[test]
    fn srgb_mipmap_averages_in_linear_light_and_configures_anisotropy() {
        // 2x2 checkerboard: two black (0) and two white (255) pixels -> linear 0.5 -> sRGB ~188
        let lvl0 = vec![
            0, 0, 0, 255, // (0, 0)
            255, 255, 255, 255, // (1, 0)
            255, 255, 255, 255, // (0, 1)
            0, 0, 0, 255, // (1, 1)
        ];
        let img = create_mipmapped_rgba8_image(
            2,
            2,
            lvl0,
            TextureFormat::Rgba8UnormSrgb,
            MipmapKind::SrgbColor,
            true,
            false,
        );
        assert_eq!(img.texture_descriptor.mip_level_count, 2);
        let data = img.data.as_ref().expect("mip chain present");
        assert_eq!(data.len(), (4 + 1) * 4);
        // Mip 1 begins at byte 16
        assert!((data[16] as i32 - 188).abs() <= 1, "got {}", data[16]);
        assert_eq!(data[19], 255);

        match &img.sampler {
            ImageSampler::Descriptor(d) => {
                assert_eq!(d.anisotropy_clamp, 8);
                assert_eq!(d.address_mode_u, ImageAddressMode::Repeat);
                assert_eq!(d.address_mode_v, ImageAddressMode::ClampToEdge);
            }
            _ => panic!("expected explicit anisotropic ImageSamplerDescriptor"),
        }
    }

    #[test]
    fn normal_mipmap_stays_unit_length_and_toksvig_lifts_high_variance_roughness() {
        // Opposing tilted normals in a 2x2 quad: (0, +X/+Z) vs (0, -X/+Z) -> average points to +Z with len < 1
        let tilta = [225_u8, 128, 225, 255];
        let tiltb = [30_u8, 128, 225, 255];
        let mut normal_varied = Vec::new();
        normal_varied.extend_from_slice(&tilta);
        normal_varied.extend_from_slice(&tiltb);
        normal_varied.extend_from_slice(&tilta);
        normal_varied.extend_from_slice(&tiltb);

        let rm_base = [0_u8, 51, 200, 255].repeat(4); // roughness ~0.20 (51/255)
        let albedo = [180_u8, 180, 180, 255].repeat(4);

        let (_, rm_varied, n_varied) = create_mipmapped_pbr_triplet(
            2,
            2,
            albedo.clone(),
            rm_base.clone(),
            normal_varied,
            true,
            true,
        );

        let flat = [128_u8, 128, 255, 255].repeat(4);
        let (_, rm_flat, _) =
            create_mipmapped_pbr_triplet(2, 2, albedo, rm_base, flat, true, true);

        let rm_v_data = rm_varied.data.as_ref().unwrap();
        let rm_f_data = rm_flat.data.as_ref().unwrap();
        let n_v_data = n_varied.data.as_ref().unwrap();
        assert_eq!(rm_v_data.len(), 20);
        assert_eq!(rm_f_data.len(), 20);
        // Flat normal keeps base roughness (~51); varied normal widens mip-1 roughness cleanly above 51.
        assert_eq!(rm_f_data[17], 51);
        assert!(
            rm_v_data[17] > rm_f_data[17] + 20,
            "expected Toksvig roughness widening: varied={} vs flat={}",
            rm_v_data[17],
            rm_f_data[17]
        );

        // Averaged normal in mip 1 (bytes 16..20) must remain close to unit length (+Z ~ 255)
        let nx = n_v_data[16] as f32 / 127.5 - 1.0;
        let ny = n_v_data[17] as f32 / 127.5 - 1.0;
        let nz = n_v_data[18] as f32 / 127.5 - 1.0;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!((len - 1.0).abs() < 0.03, "expected unit normal, got len={len}");
    }
}
