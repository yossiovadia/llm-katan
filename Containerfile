FROM registry.fedoraproject.org/fedora:44 AS builder

RUN dnf install -y python3-pip gcc python3-devel which

WORKDIR /opt/llm-katan

RUN python3 -m venv venv
ENV PATH="/opt/llm-katan/venv/bin:$PATH"

COPY . .

RUN pip install .
RUN which llm-katan

FROM registry.fedoraproject.org/fedora:44

RUN dnf install -y python3

WORKDIR /opt/llm-katan

COPY --from=builder /opt/llm-katan/venv /opt/llm-katan/venv
ENV PATH="/opt/llm-katan/venv/bin:$PATH"

ENTRYPOINT ["llm-katan"]