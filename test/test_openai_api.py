# -*- coding: utf-8 -*-
""""""

import logging
import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def main():
    client = AsyncOpenAI(
            api_key="token-abc123",
            base_url="http://localhost:8000/v1",
        )

    response = await client.completions.create(
                        model="Qwen/QwQ-32B",
                        prompt="Hello, how are you?",
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=100,
                        # stop=["<|end_search_query|>"],
                        extra_body={
                            'top_k': 1,
                            'include_stop_str_in_output': True,
                            'repetition_penalty': 1,
                            # 'min_p': min_p
                        },
                        timeout=600,
                    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())