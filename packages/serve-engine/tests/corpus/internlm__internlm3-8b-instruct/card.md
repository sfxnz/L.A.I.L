# InternLM3-8B-Instruct

Small chat model. Tools use the InternLM2 plugin format.

```shell
vllm serve internlm/internlm3-8b-instruct --trust-remote-code --enable-auto-tool-choice --tool-call-parser internlm
```
