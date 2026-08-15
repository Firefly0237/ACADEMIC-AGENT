# Milvus 配置说明

**默认情况下这个目录是空的，这本身就是结论。**

决策记录 D4 修正过一个说法：`milvus.yaml` 一度被写成「换 TEI 路线的代价」，
但官方 yaml 片段自带注释 `enable: true # enabled by default. no action required.`
——TEI provider 默认就是开的，**无需任何改动**。

## 什么时候才需要往这里放 milvus.yaml

只有一种情况：TEI 启动时带了 `--api-key`。此时 Milvus 需要拿到 credential 才能调它。

做法：

1. 从运行中的容器里导出**完整**的官方默认配置作为底稿——

   ```bash
   docker run --rm milvusdb/milvus:v2.6.0 cat /milvus/configs/milvus.yaml > milvus.yaml
   ```

2. 在底稿里补上 credential 与 function 段；
3. 在 `docker-compose.yml` 的 `milvus` 服务里加回挂载
   `./milvus/milvus.yaml:/milvus/configs/milvus.yaml:ro`。

## 为什么强调「以完整官方 yaml 为底稿」

容器内的 `/milvus/configs/milvus.yaml` 是一份完整配置，挂载是**整份替换**而不是合并。
挂一份只写了几行的片段上去，等于把其余几百项默认值一起清空，Milvus 会直接起不来
——而且报错信息指向的是某个八竿子打不着的缺省项，排查方向会被带偏。
