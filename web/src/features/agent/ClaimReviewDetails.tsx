type Review = { claims: Array<{ claim_id: string; verdict: string; explanation: string }> };
type Grid = { claims: Record<string, { quotes: Record<string, { relation: string; conditions_preserved: boolean; explanation: string }> }> };
const verdicts: Record<string,string> = { entailed: "支持关系通过", contradicted: "反驳关系通过", mixed: "存在相反证据", insufficient: "证据不足", invalid: "核验未通过" };
export function ClaimReviewDetails({ claimId, review, grid }: { claimId: string; review?: Review; grid?: Grid }) {
  const item = review?.claims.find(value => value.claim_id === claimId);
  if (!item) return null;
  const opinions = Object.values(grid?.claims[claimId]?.quotes || {});
  return <details className="claim-review">
    <summary>AI 引用核验 · {verdicts[item.verdict] || item.verdict}</summary>
    <p>{item.explanation}</p>
    {opinions.map((opinion,index) => <p key={index}>引用 {index+1}：{opinion.explanation}
      {!opinion.conditions_preserved && <strong> · 条件保留核验未通过</strong>}
    </p>)}
    <small className="muted">这是自动审查意见；原文定位与语义支持分别核验，关键结论可展开原文复核。</small>
  </details>;
}
