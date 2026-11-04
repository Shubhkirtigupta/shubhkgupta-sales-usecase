import os

from anthropic import APIError, Anthropic
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
	api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
	if not api_key:
		raise SystemExit("Set ANTHROPIC_API_KEY to your Anthropic API key first.")

	try:
		with Anthropic(api_key=api_key) as client:
			response = client.messages.create(
				model="claude-haiku-4-5",
				max_tokens=128,
				messages=[{"role": "user", "content": "Hi, what is 2+2?"}],
			)
		for block in response.content:
			if block.type == "text":
				print(block.text)
	except APIError:
		raise SystemExit(
			"Claude API request failed. Check your key, API credits, "
			"model access, and network connection."
		) from None


if __name__ == "__main__":
	main()
