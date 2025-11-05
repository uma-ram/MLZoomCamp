FROM agrigorev/zoomcamp-model:2025

# add your stuff here
# Install all the dependencies from pyproject.toml
# Copy your FastAPI script
# Run it with uvicorn

WORKDIR /app

# Install uv tool
RUN pip install uv

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY "homework.py" "pipeline_v1.bin" ./

EXPOSE 9696

ENTRYPOINT [ "uvicorn", "homework:app","--host","0.0.0.0", "--port", "9696"]