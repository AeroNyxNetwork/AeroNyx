# AeroNyx

**AeroNyx is a privacy network whose relays are blind by design: nodes carry sealed data and cannot read it.** This repository is the open-source reference implementation of the AeroNyx node and its protocol, written in Rust.

- Website: <https://aeronyx.network> · Documentation: <https://docs.aeronyx.network> · Operator console: <https://app.aeronyx.network>
- License: [GNU AGPL-3.0](LICENSE) (node software and protocol design)
- Status: **v0.1, early network.** Read [Threat model and honest limits](#threat-model-and-honest-limits) before relying on it.

## What is open source here, and what is not

| Part | Open source? |
|---|---|
| Node software (`aeronyx-server`), protocol and cryptography (`aeronyx-core`), transport, blind issuer | **Yes — AGPL-3.0, this repository** |
| Protocol design documents | **Yes** — [`docs/`](docs/) and <https://docs.aeronyx.network> |
| The AeroNyx app (iOS, Android, macOS, Windows) | **No — closed source.** It is not part of this repository. |

What you can and cannot check from here: the **node-side** guarantee — that a node holds no user keys and cannot read what it carries — can be reviewed in this repository without the app. What the app does **on your own device** (sealing messages, holding your keys) is not publicly reviewable, because the app's source is closed. Where the protocol relies on the device doing the right thing, that is a trust assumption, and we state it rather than hide it.

## What the protocol does

The node never needs plaintext to do its job. Each mechanism below is implemented in this repository:

- **Blind relay.** Nodes forward sealed envelopes between devices and see only an envelope, its size and where to hand it next — not the content, and not the sender's or recipient's keys (`aeronyx-core::protocol`, `aeronyx-server`).
- **Layered onion routing.** A message is wrapped in one encryption layer per hop (ephemeral X25519 → ECDH → HKDF-SHA256 → XChaCha20-Poly1305), so each relay can peel only its own layer and learns only the next hop. A route of **two hops** means no single node knows both ends. In the app this route is used for messages and is **opt-in today**, not the default. Relay keys rotate every 24 hours and live in memory only, for forward secrecy (`aeronyx-core/src/protocol/onion.rs`, `aeronyx-server/src/services/onion_keys.rs`).
- **Signed node discovery.** Nodes publish Ed25519-signed descriptors (including the key clients use to build onion layers). Peers verify a descriptor before treating a node as live (`aeronyx-core/src/protocol/discovery.rs`).
- **Blind admission.** Admission tokens use RFC 9474 blind signatures, issued by a separate, isolated process that holds the private key, so the issuer cannot link a token to the person it was issued to (`aeronyx-blind-issuer`, `deploy/blind-issuer`).
- **Node-blind memory storage (MemChain).** Nodes store device-encrypted records and keyed blind indexes; encryption, decryption and search happen on the user's device. Design: [`docs/memchain-node-blind-refactor.md`](docs/memchain-node-blind-refactor.md), [`docs/memchain-blind-cognition-design.md`](docs/memchain-blind-cognition-design.md).
- **VPN transport.** A UDP tunnel with a TUN device abstraction (`aeronyx-transport`). It is a single-node tunnel: like any VPN, the node you connect through can see where your traffic is going (but not the contents of encrypted connections). See [`docs/rust-vpn-transport-fallback-dev.md`](docs/rust-vpn-transport-fallback-dev.md).

Node-to-node discovery and the encrypted relay are specified in [`docs/node-discovery-and-encrypted-relay-plan.md`](docs/node-discovery-and-encrypted-relay-plan.md).

## Threat model and honest limits

We would rather state the limits than have them found for us.

- The relay layer is designed against **honest-but-curious relays**: an operator who follows the protocol but looks at everything it can see. It is **not** designed to resist a **global passive adversary** that watches the whole network at once; traffic-analysis resistance of that kind (constant-length packets, mixing) is future work.
- The onion route is **two hops** (not three), it is **opt-in** for messages, and VPN browsing does not use it. The network is **small**, so the anonymity set is small. Treat it as a meaningful reduction in who can see what, not as strong anonymity.
- There is **no formal proof** of the protocol and no claim that it removes every metadata risk. Metadata a network necessarily sees (for example, that a connection occurs, and its size and timing) is not hidden.
- Nothing here makes unlawful activity lawful. You are responsible for following the law where you are.

## Build and run

The toolchain is pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (Rust 1.97.1) so every build uses the same compiler.

```bash
# build the node
cargo build --release -p aeronyx-server

# run the test suite
cargo test --workspace
```

To run a node, start from [`deploy/node/README.md`](deploy/node/README.md) and [`deploy/node/server.example.toml`](deploy/node/server.example.toml). `deploy/node/install.sh` and `upgrade.sh` install and update a node as a systemd service. The blind issuer has its own guide in [`deploy/blind-issuer/README.md`](deploy/blind-issuer/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `crates/aeronyx-core` | Protocol definitions and cryptography |
| `crates/aeronyx-server` | The node binary and orchestration |
| `crates/aeronyx-transport` | Network I/O and TUN device abstraction |
| `crates/aeronyx-common` | Shared utilities and types |
| `crates/aeronyx-blind-issuer` | Isolated RFC 9474 blind-signing service |
| `deploy/` | Install, upgrade, health-check and systemd material |
| `docs/` | Protocol and design documents |

## License

Copyright © AeroNyx. Licensed under the **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`) — see [LICENSE](LICENSE).

In practice: you may use, study, modify and run this software. If you run a **modified** version as a network service, AGPL section 13 requires you to offer the corresponding source of your modified version to the users of that service. The "AeroNyx" name and logo are not licensed for use.

The app is a separate, closed-source work and is not covered by this license. Commits made before the relicense on 2026-10-08 declared `MIT OR Apache-2.0` in `Cargo.toml`; copies obtained before then remain available under those terms.

## Reporting a vulnerability

Please report security issues privately to **hi@aeronyx.network** instead of opening a public issue, and allow time for a fix before disclosing.

## In other languages

- **简体中文** — AeroNyx 是一个设计上「中继盲目」的隐私网络：节点只负责传递加密后的数据，读不到内容。本仓库是节点与协议的开源参考实现（Rust，AGPL-3.0）。App 不开源，不在本仓库内。当前为 v0.1 早期网络，两跳洋葱路由目前为可选项，VPN 浏览为单节点隧道，匿名集较小，局限见上文。
- **繁體中文** — AeroNyx 是一個設計上「中繼盲目」的隱私網路：節點只負責傳遞加密後的資料，讀不到內容。本倉庫是節點與協議的開源參考實作（Rust，AGPL-3.0）。App 不開源，不在本倉庫內。目前為 v0.1 早期網路，兩跳洋蔥路由目前為可選項，VPN 瀏覽為單一節點通道，匿名集較小，限制見上文。
- **日本語** — AeroNyx は、中継ノードが設計上「盲目」なプライバシーネットワークです。ノードは暗号化されたデータを運ぶだけで、中身は読めません。このリポジトリはノードとプロトコルのオープンソース参照実装（Rust、AGPL-3.0）です。アプリはクローズドソースで、ここには含まれません。現在は v0.1 の初期ネットワークで、2 ホップのオニオン経路は任意設定、VPN は単一ノードのトンネルであり、匿名性の母集団は小さく、限界は上記のとおりです。
- **한국어** — AeroNyx는 중계 노드가 설계상 "눈먼" 프라이버시 네트워크입니다. 노드는 암호화된 데이터를 전달할 뿐 내용을 읽을 수 없습니다. 이 저장소는 노드와 프로토콜의 오픈소스 참조 구현(Rust, AGPL-3.0)입니다. 앱은 비공개 소스이며 여기에 포함되지 않습니다. 현재 v0.1 초기 네트워크이며, 2홉 어니언 경로는 선택 사항이고 VPN은 단일 노드 터널이며 익명 집합이 작습니다. 한계는 위를 참고하세요.
- **Русский** — AeroNyx — приватная сеть, в которой ретрансляторы слепы по замыслу: узлы передают запечатанные данные и не могут их прочитать. Этот репозиторий — открытая эталонная реализация узла и протокола (Rust, AGPL-3.0). Приложение имеет закрытый код и в репозиторий не входит. Сейчас это ранняя сеть v0.1: двухузловой луковый маршрут пока включается по желанию, VPN — туннель через один узел, анонимное множество небольшое; ограничения описаны выше.
- **Español** — AeroNyx es una red de privacidad cuyos relés son ciegos por diseño: los nodos llevan datos sellados y no pueden leerlos. Este repositorio es la implementación de referencia de código abierto del nodo y del protocolo (Rust, AGPL-3.0). La app es de código cerrado y no forma parte de este repositorio. Hoy es una red temprana v0.1: la ruta cebolla de dos saltos es opcional, la VPN es un túnel de un solo nodo y el conjunto de anonimato es pequeño; los límites están descritos arriba.
