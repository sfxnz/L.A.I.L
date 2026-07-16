# Demo workspace

Default **L.A.I.L Workbench** project root (Composer agent sandboxes here).

## Try in Workbench

Open **Workbench** (`/workbench`) and ask:

> Explore this workspace and create `local-ai-survival-guide.md` with practical vLLM / llama.cpp setup notes for this machine.

The agent should:

1. List / read files  
2. Stream **Thought** / **Ran N commands** / **Creating**  
3. Open the new file in an editor tab  

## Notes

- Shell tools run with **cwd = this workspace**.  
- Paths cannot escape the workspace root.  
- Point **Configure → Default model** at your live vLLM (or llama.cpp) model id.  
