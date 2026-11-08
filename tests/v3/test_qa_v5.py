"""Official parent/child LangGraph with explicit model/tool/ledger doubles.

No network, PG, real academic opinions or provider requests in these controls.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from citequest.agents_v3.catalog_scope import POLICY
from citequest.agents_v3.contracts import PROFILES,AgentError,digest,graph_version_for_kind
from citequest.agents_v3.graph import AgentGraph

sys.path.insert(0,str(Path(__file__).parent))
from test_agent_workset import FixtureExecution,FixtureMemory,FixtureTools

ROOT = Path(__file__).resolve().parents[2]


class Cursor:
    def __init__(self,rows):
        self.rows = rows
    def fetchall(self):
        return self.rows
    def fetchone(self):
        return self.rows[0] if self.rows else None


class Execution(FixtureExecution):
    def __init__(self):
        super().__init__("qa")
        profile = {**PROFILES["qa-v5"].model_dump(),"catalog_scope_policy":POLICY}
        self.value.update(profile=profile,profile_hash=digest(profile),graph_version=graph_version_for_kind("qa","qa-v5"))
        self.value["brief"]["question"] = "比较Word2vec和fastText。"
        self.turns = [{"turn_id":"U1","text":self.value["brief"]["question"]}]
        self.known,self.issued,self.credits = {},{},{}
        self.database = self
        self.latest_known = None
    @contextmanager
    def connect(self):
        yield self
    def execute(self,query,args):
        if "SELECT ref_id,payload" in query:
            return Cursor([{"ref_id":key,"payload":copy.deepcopy(self.issued[key])} for key in args[1] if key in self.issued])
        if "SELECT ref_id FROM" in query:
            return Cursor([{"ref_id":key} for key in sorted(self.issued)])
        if "SELECT id,request_hash,status,usage,error_code" in query:
            row = self.known.get(args[1])
            return Cursor([{"id":row["id"],"request_hash":digest(row["request"]),"status":"settled","usage":{"input":100,"output":40,"cached":0},"error_code":None}] if row else [])
        raise AssertionError(query)
    def settled_model_request(self,key):
        row = self.known.get(key)
        return {"request":copy.deepcopy(row["request"]),"request_hash":digest(row["request"])} if row else None
    def content_revision_count(self):
        return len(self.credits)
    def reserve_content_revision(self,key,*,phase,error_codes,source_logical_key=None):
        identity = digest({"phase":phase,"error_codes":error_codes,"source_logical_key":source_logical_key})
        if key in self.credits:
            assert self.credits[key]["identity"] == identity
            return self.credits[key]
        if source_logical_key not in self.known:
            raise AgentError("content_revision_requires_known_receipt")
        if len(self.credits) >= 2:
            raise AgentError("content_revision_limit")
        self.credits[key] = {"ordinal":len(self.credits)+1,"identity":identity}
        return self.credits[key]


class Tools(FixtureTools):
    def __init__(self,execution,*,empty=False):
        super().__init__(execution,window_counts=[2])
        self.empty = empty
    async def call(self,name,args,*,logical_key,planning_context):
        await self.check_scope()
        assert set(planning_context) == {"planner_attempt_id","planner_request_sha256","candidate_sha256","plan_sha256"}
        if logical_key in self.cache:
            return copy.deepcopy(self.cache[logical_key])
        self.calls.append((name,copy.deepcopy(args),copy.deepcopy(planning_context)))
        self.execution.value["tool_calls"] += 1
        if name == "search_chunk_evidence":
            values = [] if self.empty else list(self.evidence.values())
            self.execution.issued.update({item["ref_id"]:copy.deepcopy(item) for item in values})
            result = {"status":"ok","evidence":values,"empty":not values}
        elif name == "verify_evidence_batch":
            assert set(args["ref_ids"]) <= self.execution.issued.keys()
            result = {"status":"ok","verified":len(args["ref_ids"])}
        else:
            raise AssertionError(name)
        self.cache[logical_key] = copy.deepcopy(result)
        return result


class Provider:
    def __init__(self,execution,tools,*,author_fault=None,review_fault=None,repeat=False,partial=False,clarify=False):
        self.execution,self.tools = execution,tools
        self.author_fault,self.review_fault,self.repeat,self.partial,self.clarify = author_fault,review_fault,repeat,partial,clarify
        self.requests,self.calls = [],{"plan":0,"author":0,"review":0}
    async def complete(self,payload,key):
        await self.tools.check_scope()
        if key in self.execution.known:
            assert payload == self.execution.known[key]["request"]
            return copy.deepcopy(self.execution.known[key]["response"])
        prompt = json.loads(payload["input"][0]["content"])
        role = payload["text"]["format"]["name"].removeprefix("citequest_qa5_")
        self.calls[role] += 1
        self.requests.append({"request":copy.deepcopy(payload),"key":key,"role":role})
        self.execution.value["model_calls"] += 1
        self.execution.value["input_tokens"] += 100
        self.execution.value["output_tokens"] += 40
        turn = prompt["current_user_turn"]
        if role == "plan":
            prior = prompt["current_requirements"]["active"]
            ambiguous = self.clarify and turn["turn_id"] == "U1"
            action = "supersede" if turn["turn_id"] == "U2" and prior else "add"
            ops = [] if ambiguous else [{"action":action,"target_ids":[item["id"] for item in prior] if action == "supersede" else [],
                "category":"focus" if action == "supersede" else "detail" if prior else "question",
                "spans":[{"turn_id":turn["turn_id"],"start":0,"end":len(turn["text"])}]}]
            value = {"requirement_delta":{"parent_sha256":prompt["requirements_parent_sha256"],
                "authority_sha256":prompt["Task_source_authority_sha256"],"turn_id":turn["turn_id"],"turn_sha256":turn["sha256"],
                "retain_ids":[item["id"] for item in prior] if action == "add" else [],"operations":ops,
                "intent":"needs_clarification" if ambiguous else "clear","clarification":"Please restate intended focus." if ambiguous else None},
                "action":"clarify" if ambiguous else "retrieve","queries":[] if ambiguous else [{"query":turn["text"],"limit":2}],"existing_ref_ids":[]}
        elif role == "author":
            units = [u for ctx in prompt["untrusted_evidence_data"] for u in ctx["quote_units"] if "text" in u]
            if prompt.get("untrusted_previous_candidate"):
                value = json.loads(prompt["untrusted_previous_candidate"])
                for claim in value["claims"]:
                    claim["status"] = "supported" if units else "insufficient"
            else:
                value = {"kind":"qa","requested_direction":"neutral","claims":[{"id":"C1","statement":"Explicit synthetic answer.",
                    "status":"supported" if units else "insufficient","edges":[{"quote_id":units[0]["quote_id"],"relation":"supports",
                        "rationale":"Explicit synthetic opinion.","conditions":[]}] if units else [],"limitations":["Synthetic fixture, not academic truth."]}],
                    "gaps":[],"evidence_status":"supported" if units else "insufficient","retrieval_scope":"Explicit two fixture chunks."}
            if self.author_fault == "enum" and (self.repeat or self.calls[role] == 1):
                value["claims"][0]["status"] = "context_only"
        else:
            assert not prompt["selected_user_skills"]
            statuses = {c["id"]:c["status"] for c in prompt["candidate"]["claims"]}
            value = {"candidate_sha256":prompt["requirements_candidate_sha256"],"user_turn_sha256":turn["sha256"],
                "opinions":[{"requirement_id":i,"verdict":"correct","rationale":"Explicit synthetic opinion."} for i in prompt["required_requirement_review_ids"]],
                "current_turn_coverage":"complete","decision":"accepted","claims":{
                    cid:{"verdict":"insufficient" if statuses[cid] == "insufficient" else "entailed","conditions_preserved":True,
                        "explanation":"Synthetic opinion.","quotes":{qid:{"relation":"matches_declared_relations","conditions_preserved":True,
                            "explanation":"Synthetic opinion."} for qid in qids}} for cid,qids in prompt["review_grid"]["claim_quote_ids"].items()},
                "source_instruction_injection_detected":False,"requested_stance_bias_detected":False,
                "request_coverage":"partial" if self.partial else "complete","uncovered_requests":["Synthetic omission."] if self.partial else []}
            fault = self.review_fault if self.repeat or self.calls[role] == 1 else None
            if fault == "missing":
                value["claims"].pop("C1")
            elif fault == "semantic":
                value["claims"]["C1"]["verdict"] = "invalid"
            elif fault == "goal":
                value["decision"] = "needs_revision"
        result = {"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":json.dumps(value)}]}]}
        self.execution.known[key] = {"id":uuid4(),"request":copy.deepcopy(payload),"response":copy.deepcopy(result)}
        return result


class QA5Controls(unittest.IsolatedAsyncioTestCase):
    async def run_case(self,*,execution=None,saver=None,tools=None,**modes):
        execution = execution or Execution()
        tools = tools or Tools(execution)
        provider = Provider(execution,tools,**modes)
        with patch("citequest.agents_v3.graph.PostgresSaver",return_value=saver or InMemorySaver()),patch.object(AgentGraph,"user_turns",lambda _:execution.turns):
            graph = AgentGraph(execution,provider,tools,FixtureMemory(),ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(graph.run)
        return execution,provider,tools,result

    async def test_initial_actual_parent_child_Q_grid_and_provenance(self):
        execution,provider,tools,result = await self.run_case()
        self.assertEqual(provider.calls,{"plan":1,"author":1,"review":1})
        self.assertEqual(result["requirements_state"]["revision"],1)
        self.assertEqual(result["result"]["publication_status"],"validated_ai_review")
        self.assertEqual(execution.finishes[-1]["state"],"succeeded")
        self.assertEqual(len(tools.calls),3)

    async def test_three_turns_supersede_scope_then_preserve_new_detail_same_Task(self):
        execution,saver = Execution(),InMemorySaver()
        tools = Tools(execution)
        _,_,_,first = await self.run_case(execution=execution,saver=saver,tools=tools)
        for seq,text in ((2,"不要比较Word2vec，只看fastText。"),(3,"保留fastText范围，再说明数值缺口。")):
            execution.command = {"id":uuid4(),"kind":"followup"}
            execution.value["brief"]["question"] = text
            execution.value["revision"] += 1
            execution.turns.append({"turn_id":"U"+str(seq),"text":text})
            _,provider,_,result = await self.run_case(execution=execution,saver=saver,tools=tools)
            author = json.loads(next(r["request"] for r in provider.requests if r["role"] == "author")["input"][0]["content"])
            self.assertNotIn("比较Word2vec和fastText。",str(author["provisional_requirements"]))
        self.assertEqual(result["requirements_state"]["revision"],3)
        self.assertEqual([r["id"] for r in result["requirements_state"]["requirements"] if r["state"] == "active"],["R2","R3"])
        self.assertEqual(execution.value["model_calls"],9)
        self.assertEqual(execution.content_revision_count(),0)
        self.assertEqual(first["requirements_state"]["requirements"][0]["state"],"active")

    async def test_author_and_review_formats_share_two_credit_receipts(self):
        execution,provider,_,_ = await self.run_case(author_fault="enum",review_fault="missing")
        self.assertEqual(execution.content_revision_count(),2)
        self.assertEqual(execution.value["model_calls"],5)

    async def test_semantic_rejection_directed_revision_independent_review_again(self):
        execution,provider,_,_ = await self.run_case(review_fault="semantic")
        self.assertEqual(execution.content_revision_count(),1)
        self.assertEqual(provider.calls["review"],2)

    async def test_goal_rejection_replans_before_committing(self):
        execution,provider,_,result = await self.run_case(review_fault="goal")
        self.assertEqual(execution.content_revision_count(),1)
        self.assertEqual(provider.calls["plan"],2)
        self.assertEqual(result["requirements_state"]["revision"],1)

    async def test_partial_and_empty_do_not_claim_complete_task_success(self):
        execution,_,_,result = await self.run_case(partial=True)
        self.assertEqual(execution.finishes[-1]["state"],"partial")
        self.assertEqual(result["result"]["completion_status"],"partial_request_coverage")
        execution = Execution()
        execution,_,_,result = await self.run_case(execution=execution,tools=Tools(execution,empty=True))
        self.assertEqual(execution.finishes[-1]["state"],"partial")
        self.assertFalse(result["result"]["citations"])

    async def test_third_content_repair_fails_without_promoting_requirements(self):
        execution = Execution()
        with self.assertRaises(AgentError):
            await self.run_case(execution=execution,author_fault="enum",repeat=True)
        self.assertEqual(execution.content_revision_count(),2)
        self.assertEqual(execution.value["model_calls"],4)

    async def test_cold_graph_after_known_author_replays_without_new_model(self):
        execution,saver = Execution(),InMemorySaver()
        tools,original = Tools(execution),AgentGraph.observed_node
        died = {"value":False}
        def observed(graph,phase,node):
            wrapped = original(graph,phase,node)
            if phase != "qa5_author":
                return wrapped
            def stop(state):
                result = wrapped(state)
                if not died["value"]:
                    died["value"] = True
                    raise RuntimeError("synthetic_death_after_known_author")
                return result
            return stop
        with patch.object(AgentGraph,"observed_node",observed),self.assertRaises(RuntimeError):
            await self.run_case(execution=execution,saver=saver,tools=tools)
        self.assertEqual(execution.value["model_calls"],2)
        _,provider,_,result = await self.run_case(execution=execution,saver=saver,tools=tools)
        self.assertEqual(execution.value["model_calls"],3)
        self.assertEqual(provider.calls["author"],0)
        self.assertEqual(result["requirements_state"]["revision"],1)

    async def test_clarification_resumes_same_Task_with_new_Brief_anchor_and_no_credit_reset(self):
        execution,saver = Execution(),InMemorySaver()
        tools = Tools(execution)
        _,_,_,first = await self.run_case(execution=execution,saver=saver,tools=tools,clarify=True)
        self.assertTrue(first.get("__interrupt__"))
        self.assertEqual(execution.value["model_calls"],1)
        text = "只看fastText，说明实验条件。"
        execution.command = {"id":uuid4(),"kind":"resume","payload":{"message":text}}
        execution.value["brief"]["question"] = text
        execution.value["revision"] += 1
        execution.turns.append({"turn_id":"U2","text":text})
        _,_,_,result = await self.run_case(execution=execution,saver=saver,tools=tools,clarify=True)
        self.assertEqual(execution.value["model_calls"],4)
        self.assertEqual(execution.content_revision_count(),0)
        self.assertEqual(result["requirements_state"]["revision"],1)
        self.assertEqual([row["turn_id"] for row in result["requirements_state"]["turns"]],["U2"])
        self.assertEqual(execution.finishes[-1]["state"],"succeeded")
