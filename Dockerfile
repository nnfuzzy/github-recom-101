FROM ubuntu:latest

# System packages 
RUN apt-get update && apt-get install -y curl

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda install -y mamba -n base -c conda-forge

COPY . /home/
RUN ls -lhrt  /
RUN ls -lhrt  /home/

WORKDIR /home

ENV DATA_PATH="data/"

RUN conda env update -f simple.yml --prune
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "recom101", "/bin/bash", "-c"]


EXPOSE 8501


# Run the executable
ENTRYPOINT ["/miniconda/envs/recom101/bin/streamlit", "run"]
CMD ["/home/recom_app.py"]
