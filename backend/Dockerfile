FROM python:3.11

# Set working directory to /app
WORKDIR /app

# Copy the backend directory
COPY . /app

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib \
    && rm ta-lib-0.4.0-src.tar.gz

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to keep the container running
CMD ["tail", "-f", "/dev/null"]