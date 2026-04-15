import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tracers.langchain import wait_for_all_tracers

from chains.pipeline import run_pipeline, build_llm, build_debug_chain

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def read_file(filename: str) -> str:
    return (DATA_DIR / filename).read_text(encoding="utf-8")


def main():
    load_dotenv()

    llm = build_llm()
    job_description = read_file("job_description.txt")

    resumes = {
        "strong": read_file("resume_strong.txt"),
        "average": read_file("resume_average.txt"),
        "weak": read_file("resume_weak.txt"),
    }

    final_results = {}

    for label, resume_text in resumes.items():
        result = run_pipeline(
            resume_text,
            job_description,
            llm,
            langsmith_extra={
                "tags": [label],
                "metadata": {"candidate_bucket": label}
            }
        )
        final_results[label] = result

        print(f"\n{'=' * 20} {label.upper()} CANDIDATE {'=' * 20}")
        print(json.dumps(result, indent=2))

    debug_chain = build_debug_chain(llm)
    incorrect_output = debug_chain.invoke(
        {"resume": resumes["weak"]},
        config={
            "run_name": "weak_resume_debug_run",
            "tags": ["weak", "debug"],
            "metadata": {"purpose": "show-incorrect-output"}
        }
    )

    print(f"\n{'=' * 20} DEBUG INCORRECT OUTPUT {'=' * 20}")
    print(incorrect_output.content if hasattr(incorrect_output, "content") else incorrect_output)

    with open(BASE_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    wait_for_all_tracers()
    print("\nAll traces flushed to LangSmith.")
    print("Results saved to results.json")


if __name__ == "__main__":
    main()