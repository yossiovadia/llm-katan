FROM registry.fedoraproject.org/fedora:44 AS builder

RUN dnf install -y python3-pip gcc python3-devel

WORKDIR /opt/llm-katan

RUN python3 -m venv venv
ENV PATH="/opt/llm-katan/venv/bin:$PATH"

COPY pyproject.toml .
COPY llm_katan ./llm_katan
RUN pip install .

FROM registry.fedoraproject.org/fedora:44

EXPOSE 8000 443

RUN dnf install -y python3

WORKDIR /opt/llm-katan

COPY --from=builder /opt/llm-katan/ /opt/llm-katan/
ENV PATH="/opt/llm-katan/venv/bin:$PATH"

COPY . .

ENTRYPOINT ["llm-katan"]
