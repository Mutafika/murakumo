//! エフェクト — マテリアルに付随するアニメーション/トランジション
//!
//! 将来的にパーティクル、波紋、ワープなどを追加。

/// エフェクトの状態
#[derive(Clone, Copy, Debug)]
pub enum EffectState {
    Idle,
    Playing { progress: f32 },
    Done,
}

impl EffectState {
    pub fn tick(&mut self, dt: f32, speed: f32) {
        if let EffectState::Playing { progress } = self {
            *progress = (*progress + dt * speed).min(1.0);
            if *progress >= 1.0 {
                *self = EffectState::Done;
            }
        }
    }

    pub fn start(&mut self) {
        *self = EffectState::Playing { progress: 0.001 };
    }

    pub fn progress(&self) -> f32 {
        match self {
            EffectState::Idle => 0.0,
            EffectState::Playing { progress } => *progress,
            EffectState::Done => 1.0,
        }
    }
}
