pub mod aurora;
pub mod bubble;
pub mod cloud;
pub mod crystal;
pub mod dissolve;
pub mod explosion;
pub mod fire;
pub mod glass;
pub mod grid;
pub mod hologram;
pub mod ice;
pub mod lava;
pub mod lightning;
pub mod metal;
pub mod neon;
pub mod portal;
pub mod rock;
pub mod shield;
pub mod skin;
pub mod smoke;
pub mod tornado;
pub mod water;

/// マテリアルが提供するもの
pub trait Material {
    /// WGSLシェーダーソースを返す
    fn shader_source(&self) -> &str;
    /// マテリアル固有のユニフォームバイト列
    fn uniform_bytes(&self) -> Vec<u8>;
}
