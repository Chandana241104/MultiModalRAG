import unittest

from ui_actions import handle_ui_event


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve_answer_chunks(self, query, **kwargs):
        return self.chunks


class FakeLLM:
    def generate_answer_with_citations(self, chunks, question):
        return (
            "Revenue is 25%",
            [{"page": "2", "type": "text", "text": "Revenue is 25%"}],
            90,
        )


class UserQueryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.session_state = SessionState(
            active_document={"filename": "source.pdf"},
            chat_history=[],
            retrieval_metrics=[],
            folio_last_context=[],
            folio_last_retrieval=None,
            folio_notice=None,
        )
        self.retriever = FakeRetriever([
            "[Page 2 | text] Revenue is 25%",
            "[Page 3 | text] Other information",
        ])
        self.settings = {
            "top_k": 5,
            "rerank": False,
            "deduplicate": False,
            "show_citations": True,
            "confidence_threshold": 0.3,
        }
        handle_ui_event(
            {"action": "ask", "question": "What is the revenue?"},
            self.session_state,
            self.retriever,
            FakeLLM(),
            self.settings,
        )

    def test_user_query_stores_answer_chunks_and_source_pages(self):
        chat = self.session_state.chat_history[0]

        self.assertEqual(chat["question"], "What is the revenue?")
        self.assertEqual(chat["answer"], "Revenue is 25%")
        self.assertEqual(len(chat["retrieved_chunks"]), 2)
        self.assertEqual(chat["source_pages"], ["2", "3"])

    def test_user_query_without_retrieval_is_still_saved(self):
        state = SessionState(
            active_document={"filename": "source.pdf"},
            chat_history=[],
            retrieval_metrics=[],
            folio_last_context=[],
            folio_last_retrieval=None,
            folio_notice=None,
        )
        handle_ui_event(
            {"action": "ask", "question": "Question with no matches"},
            state,
            FakeRetriever([]),
            FakeLLM(),
            self.settings,
        )

        self.assertEqual(len(state.chat_history), 1)
        self.assertEqual(state.chat_history[0]["retrieved_chunks"], [])
        self.assertEqual(state.chat_history[0]["source_pages"], [])
        self.assertEqual(state.chat_history[0]["evaluation"], {})

        handle_ui_event(
            {
                "action": "evaluate_user_query",
                "query_index": 0,
                "reference_answer": "Any expected answer",
                "expected_pages": "",
                "relevance_reviewed": False,
                "relevant_chunk_indices": [],
            },
            state,
            FakeRetriever([]),
            FakeLLM(),
            self.settings,
        )

        self.assertIsNone(state.chat_history[0]["evaluation"]["answer_metrics"])
        self.assertEqual(state.folio_notice["kind"], "warning")

    def test_user_query_evaluation_scores_annotated_retrieval_and_reference(self):
        handle_ui_event(
            {
                "action": "evaluate_user_query",
                "query_index": 0,
                "reference_answer": "Revenue is 25%",
                "expected_pages": "2",
                "relevance_reviewed": True,
                "relevant_chunk_indices": [0],
            },
            self.session_state,
            self.retriever,
            FakeLLM(),
            self.settings,
        )

        evaluation = self.session_state.chat_history[0]["evaluation"]
        self.assertEqual(evaluation["retrieval_precision"], 50)
        self.assertEqual(evaluation["page_precision"], 50)
        self.assertTrue(evaluation["answer_metrics"]["exact_match"])
        self.assertEqual(evaluation["answer_metrics"]["key_numbers_matched"], 1)
        self.assertEqual(evaluation["answer_metrics"]["jaccard_similarity"], 1)
        self.assertEqual(evaluation["answer_metrics"]["combined_score"], 100)

    def test_answer_scores_remain_pending_without_reference_answer(self):
        handle_ui_event(
            {
                "action": "evaluate_user_query",
                "query_index": 0,
                "reference_answer": "",
                "expected_pages": "",
                "relevance_reviewed": False,
                "relevant_chunk_indices": [],
            },
            self.session_state,
            self.retriever,
            FakeLLM(),
            self.settings,
        )

        evaluation = self.session_state.chat_history[0]["evaluation"]
        self.assertIsNone(evaluation["answer_metrics"])
        self.assertIsNone(evaluation["retrieval_precision"])
        self.assertIsNone(evaluation["page_precision"])
        self.assertEqual(self.session_state.folio_notice["kind"], "warning")


if __name__ == "__main__":
    unittest.main()
