# 应用接口协议

本目录保存基础业务API与来源引用的数据定义。

| 文件 | 职责 |
| --- | --- |
| `openapi-s03.json` | FastAPI基础业务接口的OpenAPI描述 |
| `source-reference.schema.json` | 文献与材料来源引用类型 |

接口覆盖应用数据、文件和分页等基础操作，来源引用连接文献元数据、原文与业务对象。

后端实现位于`src/citequest/v2/`，Web入口位于`src/citequest/web_v3/`。

系统关系见[架构](../../docs/architecture.md)。
