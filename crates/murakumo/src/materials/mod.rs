pub mod bubble;
pub mod glass;
pub mod portal;
pub mod grid;

/// マテリアルが提供するもの
pub trait Material {
    /// WGSLシェーダーソースを返す
    fn shader_source(&self) -> &str;
    /// マテリアル固有のユニフォームバイト列
    fn uniform_bytes(&self) -> Vec<u8>;
}
