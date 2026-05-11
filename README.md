# Murakumo（叢雲）

[seimei](https://github.com/Mutafika/seimei) 上のマテリアル / エフェクト集。

## Status

🚧 Work in progress — API は予告なく変更される可能性があります。

## Crates

- `murakumo` — マテリアル / エフェクト本体（publish 対象）
- `murakumo-gallery` — デモ・展示用バイナリ（publish 対象外）

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

`murakumo-gallery` は seimei / sabitori にも依存しているデモ用バイナリで、
ビルドが重いため `default-members` で通常ビルドからは除外しています。

gallery をビルドする場合は明示的に指定:

```sh
cargo build -p murakumo-gallery
```

seimei / sabitori を手元で編集しながら Murakumo を開発する場合は、
ルート `Cargo.toml` の `[patch]` セクションのコメントを外してください。
`../seimei`, `../sabitori` にそれぞれリポジトリが clone されている前提です。

```
myapp/
├── seimei/      # https://github.com/Mutafika/seimei
├── sabitori/    # https://github.com/Mutafika/sabitori
└── murakumo/
```

## License

MIT
