import "dotenv/config";
import { createAgent, initChatModel } from "langchain";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import express from "express";

const SYSTEM_PROMPT = `You are an AI agent specialized in mathematical reasoning.

Available tools:
- addition(num1, num2)
- multiplication(num1, num2)
- divide(num1, num2)

Workflow:
1. Analyze the user's request.
2. Identify the mathematical operations required.
3. Use tools for every arithmetic operation.
4. Chain multiple tool calls when necessary.
5. Use tool outputs as inputs for subsequent calculations.
6. Produce a clear final answer.

Important:
- Never perform arithmetic in your own reasoning.
- Always rely on tools for numerical computation.
- Handle invalid operations gracefully.
- Keep responses concise unless the user requests detailed explanations.`;

const app = express();
app.use(express.json());

app.post("/ai", async (req, res) => {
  const message = req?.body?.message;

  if (typeof message !== "string" || message.trim() === "") {
    return res.status(400).json({
      error: "Missing or invalid request body",
      details: 'Send JSON like: {"message":"What is 2 + 3?"}',
    });
  }

  const addition = tool(
    ({ num1, num2 }) => {
      return num1 + num2;
    },
    {
      name: "addition",
      description: "Add two numbers together",
      schema: z.object({ num1: z.number(), num2: z.number() }),
    },
  );

  const multiplication = tool(
    ({ num1, num2 }) => {
      return num1 * num2;
    },
    {
      name: "multiplication",
      description: "Multiply num1 with num2",
      schema: z.object({ num1: z.number(), num2: z.number() }),
    },
  );

  const divide = tool(
    ({ num1, num2 }) => {
      return num1 / num2;
    },
    {
      name: "divide",
      description: "Divide num1 by num2",
      schema: z.object({ num1: z.number(), num2: z.number() }),
    },
  );

  const model = await initChatModel("llama-3.3-70b-versatile", {
    modelProvider: "groq",
    temperature: 0.5,
    timeout: 600_000,
    maxTokens: 10000,
    streaming: true,
    apiKey: process.env.GROQ_API_KEY,
  });

  const agent = createAgent({
    model,
    tools: [addition, multiplication, divide],
    systemPrompt: SYSTEM_PROMPT,
    // checkpointer,
  });

  const agentResult = await agent.invoke(
    {
      messages: [
        {
          role: "user",
          content: message,
        },
      ],
    },
    // { configurable: { thread_id: "great-gatsby-lc" } },
  );

  const agentMessages = agentResult.messages;
  console.log(agentMessages);
  return res.status(200).json(agentMessages);
});

app.use((err, req, res, next) => {
  if (err?.type === "entity.parse.failed") {
    return res.status(400).json({
      error: "Invalid JSON",
      details:
        "The request body could not be parsed. Make sure it is valid JSON, for example: {\"message\":\"What is 2 + 3?\"}",
    });
  }

  return next(err);
});

app.listen(3000, () => {
  console.log(`server started on port 3000`);
});
