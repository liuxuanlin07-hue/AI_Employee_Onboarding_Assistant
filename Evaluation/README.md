# Evaluation - Evals Explainer

## Purpose

This folder contains the evaluation dataset and result files used to test the AI Employee Onboarding Assistant.

The evaluation is designed to measure two different parts of the system:

1. Whether the correct HR document is retrieved
2. Whether the final generated answer is correct

These two stages are evaluated separately because correct retrieval does not always guarantee a correct final answer.

---

## Evaluation Dataset

The main evaluation file is:

questions.csv

It contains 40 fixed employee questions.

The questions cover three categories:

Leave

Benefits

IT Support

The questions were written manually based on the HR documents.

They were frozen before the final evaluation runs.

The model did not generate its own evaluation questions.

This reduces the risk of evaluating the system using questions that are too similar to model-generated training or test examples.

---

## Question Design

The evaluation includes both direct questions and paraphrased questions.

Example of direct wording:

How many days of annual leave do full-time employees receive?

Example of paraphrased wording:

How much holiday can I take each year?

This is useful because employees may use different words from the words used in official company documents.

---

## Ground Truth

Each question includes a ground-truth answer.

The ground truth was created directly from the information contained in the HR documents.

The ground-truth answers are used to compare the system output with the expected result.

---

## Keyword Search Baseline

The first evaluation method uses plain keyword matching.

The same 40 questions are used.

The keyword baseline tests whether a simple lexical search can identify the correct source document.

Result:

Keyword Retrieval Accuracy = 92.5%

37 out of 40 questions retrieved the correct document.

Three paraphrased leave questions retrieved the wrong document.

This shows that keyword search works well for direct wording but is weaker when the employee uses different words with similar meanings.

---

## RAG Retrieval Evaluation

The RAG retrieval system uses embeddings and FAISS semantic search.

Each HR document is converted into an embedding.

Each employee question is also converted into an embedding.

FAISS then retrieves the closest matching document.

Result:

RAG Retrieval Accuracy = 100.0%

40 out of 40 questions retrieved the correct source document.

This indicates that semantic retrieval handled the paraphrased questions better than the keyword-search baseline.

---

## Final Answer Evaluation

The RAG chatbot also generates a final natural-language answer using the retrieved document.

Final answers were graded using the following scoring rule:

Correct = 1

Partly Correct = 0.5

Incorrect = 0

Result:

RAG Final Answer Score = 97.5%

39 answers were graded Correct.

0 answers were graded Partly Correct.

1 answer was graded Incorrect.

---

## Important Failure Example

Question 25 asked whether dental treatment was fully unlimited under the employee benefit plan.

The correct Employee Benefits document was successfully retrieved.

The document clearly states that eligible dental claims are limited to SGD 500 per year.

However, the final answer incorrectly returned the fallback response instead of the SGD 500 limit.

This is an important result because it shows:

Correct retrieval does not guarantee correct generation.

For this reason, retrieval accuracy and final-answer accuracy are reported separately.

---

## Metrics Summary

| Metric | Result |
|---|---:|
| Keyword Retrieval Accuracy | 92.5% (37/40) |
| RAG Retrieval Accuracy | 100.0% (40/40) |
| RAG Final Answer Score | 97.5% |

---

## Metrics Target

The original project target was at least 85% answer accuracy.

The prototype exceeded this target.

However, these results should be interpreted carefully because the evaluation dataset contains only 40 questions and three short simulated documents.

The results show strong performance within the defined prototype scope.

They do not prove that the same performance will continue with hundreds of documents, more ambiguous employee questions, conflicting policies, or real company data.

---

## Evaluation Files

questions.csv

Contains the 40 fixed evaluation questions and ground-truth answers.

keyword_results.csv

Contains the keyword-search retrieval results.

rag_retrieval_results.csv

Contains the semantic RAG retrieval results and retrieval distance.

rag_answers.csv

Contains the final generated answers and the source documents used.

rag_scores.csv

Contains the final answer scoring results.

evaluation_summary.csv

Contains the main evaluation metrics used in the report.

---

## Evaluation Limitations

The current evaluation has several limitations.

The dataset is small.

The HR documents are simulated.

Only three document categories are tested.

The evaluation does not test long multi-turn conversations.

It does not test conflicting document versions.

It does not test user permissions.

It does not test adversarial prompts.

It does not measure production latency or cost under high traffic.

Future evaluations should use a larger dataset, more difficult questions, multiple document versions, permission-sensitive cases, and production monitoring.
