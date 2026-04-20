pub mod aurora;
pub mod bubble;
pub mod crystal;
pub mod dissolve;
pub mod fire;
pub mod glass;
pub mod grid;
pub mod hologram;
pub mod lightning;
pub mod metal;
pub mod neon;
pub mod portal;
pub mod shield;
pub mod smoke;
pub mod warp;
pub mod water;

/// マテリアルが提供するもの
pub trait Material {
    /// WGSLシェーダーソースを返す
    fn shader_source(&self) -> &str;
    /// マテリアル固有のユニフォームバイト列
    fn uniform_bytes(&self) -> Vec<u8>;
}
