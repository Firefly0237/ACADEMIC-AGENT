"""Explicit synthetic source messages, never counted as anonymous source success."""

from html import escape


def atom(rows, *, start=0, total=None):
    entries = []
    for row in rows:
        identifier, updated, version = row[:3]
        comment = row[3] if len(row) > 3 else ""
        title = row[4] if len(row) > 4 else "Synthetic protocol control " + identifier
        entries.append(f"""<entry><id>https://arxiv.org/abs/{identifier}v{version}</id><title>{escape(title)}</title>
        <summary>Explicit synthetic abstract for automated controls.</summary><updated>{updated}</updated><published>2026-01-01T00:00:00Z</published>
        <author><name>Synthetic Control</name></author><arxiv:comment>{escape(comment)}</arxiv:comment><category term="cs.AI"/></entry>""")
    return (f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:os="http://a9.com/-/spec/opensearch/1.1/">'
            f'<os:totalResults>{len(rows) if total is None else total}</os:totalResults><os:startIndex>{start}</os:startIndex>' + ''.join(entries) + '</feed>').encode()


def oai(identifier, *, version=1, deleted=False, comment="", error=None, verb="GetRecord", token=None):
    if error:
        body = f'<error code="{error}">Explicit synthetic control</error>'
    else:
        metadata = '' if deleted else f'<metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/"><id>{identifier}</id><version version="v{version}"/><title>Synthetic OAI control</title><abstract>Synthetic OAI abstract</abstract><comments>{escape(comment)}</comments></arXivRaw></metadata>'
        body = f'<{verb}><record><header{(" status=\"deleted\"" if deleted else "")}><identifier>oai:arXiv.org:{identifier}</identifier><datestamp>2026-11-01</datestamp></header>{metadata}</record>'
        if token:
            body += f'<resumptionToken expirationDate="2026-11-02T00:00:00Z">{escape(token)}</resumptionToken>'
        body += f'</{verb}>'
    return ('<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><responseDate>2026-11-01T00:00:00Z</responseDate>' + body + '</OAI-PMH>').encode()
