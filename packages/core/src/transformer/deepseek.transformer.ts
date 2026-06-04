import { UnifiedChatRequest } from "../types/llm";
import { Transformer } from "../types/transformer";

// Module-level cache for reasoning_content across multi-turn conversations
let _reasoningCache = "";

export class DeepseekTransformer implements Transformer {
  name = "deepseek";

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider?: any
  ): Promise<UnifiedChatRequest> {
    // Limit max_tokens to DeepSeek's limit
    if (request.max_tokens && request.max_tokens > 8192) {
      request.max_tokens = 8192;
    }

    // Only apply DeepSeek-specific thinking mode fixes for DeepSeek models
    const isDeepseek =
      provider?.name?.toLowerCase().includes("deepseek") ||
      request.model?.toLowerCase().includes("deepseek");
    if (!isDeepseek) return request;

    // Inject reasoning_content from thinking.content or cache into assistant messages
    request.messages?.forEach((msg) => {
      if (msg.role === "assistant") {
        if (msg.thinking?.content) {
          msg.reasoning_content = msg.thinking.content;
          _reasoningCache = msg.thinking.content;
        } else if (_reasoningCache && !msg.reasoning_content) {
          msg.reasoning_content = _reasoningCache;
        }
      }
    });

    // Enable DeepSeek thinking mode with max effort (default for Claude Code)
    request.thinking = { type: "enabled" };
    // Clean up any residual reasoning field from upstream transformers
    delete (request as any).reasoning;

    return request;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    if (response.headers.get("Content-Type")?.includes("application/json")) {
      const jsonResponse = await response.json();
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } else if (response.headers.get("Content-Type")?.includes("stream")) {
      if (!response.body) return response;

      const decoder = new TextDecoder();
      const encoder = new TextEncoder();
      let reasoningContent = "";
      let isReasoningComplete = false;
      let buffer = "";

      const stream = new ReadableStream({
        async start(controller) {
          const reader = response.body!.getReader();
          const processBuffer = (
            buffer: string,
            controller: ReadableStreamDefaultController,
            encoder: TextEncoder
          ) => {
            const lines = buffer.split("\n");
            for (const line of lines) {
              if (line.trim()) {
                controller.enqueue(encoder.encode(line + "\n"));
              }
            }
          };

          const processLine = (
            line: string,
            context: {
              controller: ReadableStreamDefaultController;
              encoder: TextEncoder;
              reasoningContent: () => string;
              appendReasoningContent: (content: string) => void;
              isReasoningComplete: () => boolean;
              setReasoningComplete: (val: boolean) => void;
            }
          ) => {
            const { controller, encoder } = context;

            if (
              line.startsWith("data: ") &&
              line.trim() !== "data: [DONE]"
            ) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.choices?.[0]?.delta?.reasoning_content) {
                  context.appendReasoningContent(
                    data.choices[0].delta.reasoning_content
                  );
                  const thinkingChunk = {
                    ...data,
                    choices: [
                      {
                        ...data.choices[0],
                        delta: {
                          ...data.choices[0].delta,
                          thinking: {
                            content: data.choices[0].delta.reasoning_content,
                          },
                        },
                      },
                    ],
                  };
                  delete thinkingChunk.choices[0].delta.reasoning_content;
                  const thinkingLine = `data: ${JSON.stringify(
                    thinkingChunk
                  )}\n\n`;
                  controller.enqueue(encoder.encode(thinkingLine));
                  return;
                }

                if (
                  data.choices?.[0]?.delta?.content &&
                  context.reasoningContent() &&
                  !context.isReasoningComplete()
                ) {
                  context.setReasoningComplete(true);
                  // Cache the reasoning content for next multi-turn request
                  _reasoningCache = context.reasoningContent();
                  const signature = Date.now().toString();

                  const thinkingChunk = {
                    ...data,
                    choices: [
                      {
                        ...data.choices[0],
                        delta: {
                          ...data.choices[0].delta,
                          content: null,
                          thinking: {
                            content: context.reasoningContent(),
                            signature: signature,
                          },
                        },
                      },
                    ],
                  };
                  delete thinkingChunk.choices[0].delta.reasoning_content;
                  const thinkingLine = `data: ${JSON.stringify(
                    thinkingChunk
                  )}\n\n`;
                  controller.enqueue(encoder.encode(thinkingLine));
                }

                if (data.choices[0]?.delta?.reasoning_content) {
                  delete data.choices[0].delta.reasoning_content;
                }

                if (
                  data.choices?.[0]?.delta &&
                  Object.keys(data.choices[0].delta).length > 0
                ) {
                  if (context.isReasoningComplete()) {
                    data.choices[0].index++;
                  }
                  const modifiedLine = `data: ${JSON.stringify(data)}\n\n`;
                  controller.enqueue(encoder.encode(modifiedLine));
                }
              } catch (e) {
                controller.enqueue(encoder.encode(line + "\n"));
              }
            } else {
              controller.enqueue(encoder.encode(line + "\n"));
            }
          };

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                if (buffer.trim()) {
                  processBuffer(buffer, controller, encoder);
                }
                break;
              }

              const chunk = decoder.decode(value, { stream: true });
              buffer += chunk;

              const lines = buffer.split("\n");
              buffer = lines.pop() || "";

              for (const line of lines) {
                if (!line.trim()) continue;

                try {
                  processLine(line, {
                    controller,
                    encoder,
                    reasoningContent: () => reasoningContent,
                    appendReasoningContent: (content) =>
                      (reasoningContent += content),
                    isReasoningComplete: () => isReasoningComplete,
                    setReasoningComplete: (val) =>
                      (isReasoningComplete = val),
                  });
                } catch (error) {
                  console.error("Error processing line:", line, error);
                  controller.enqueue(encoder.encode(line + "\n"));
                }
              }
            }
          } catch (error) {
            console.error("Stream error:", error);
            controller.error(error);
          } finally {
            try {
              reader.releaseLock();
            } catch (e) {
              console.error("Error releasing reader lock:", e);
            }
            controller.close();
          }
        },
      });

      return new Response(stream, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          "Content-Type":
            response.headers.get("Content-Type") || "text/plain",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    return response;
  }
}
