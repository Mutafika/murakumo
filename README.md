# Murakumo（叢雲）

[seimei](https://github.com/Mutafika/seimei) 上のマテリアル / エフェクト集。

## Status

🚧 Work in progress — API は予告なく変更される可能性があります。

## Crates

- `murakumo` — マテリアル / エフェクト本体（publish 対象）
- `murakumo-gallery` — デモ・展示用バイナリ（publish 対象外、ローカル開発専用）

## Usage

`Cargo.toml` に git 依存として追加してください:

```toml
[dependencies]
murakumo = { git = "https://github.com/Mutafika/murakumo", branch = "main" }
```

再現性が必要なら `rev` を固定:

```toml
[dependencies]
murakumo = { git = "https://github.com/Mutafika/murakumo", rev = "<commit-hash>" }
```

## Local Development

`murakumo-gallery` は手元での開発・確認用バイナリで、隣接ディレクトリの
`seimei` / `sabitori` リポジトリに path 依存しています。`default-members`
で通常ビルドから除外しているので、外部利用者は意識する必要はありません。

```
myapp/
├── seimei/      # https://github.com/Mutafika/seimei
├── sabitori/    # https://github.com/Mutafika/sabitori
└── murakumo/
```

seimei を手元で編集しながら Murakumo を開発する場合は、ルート `Cargo.toml`
の `[patch]` セクションのコメントを外してください。

gallery をビルドする場合は明示的に指定:

```
cargo build -p murakumo-gallery
```

## License

MIT
