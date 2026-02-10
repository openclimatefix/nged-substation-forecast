import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    from dotenv import load_dotenv
    import obstore
    import os
    from typing import Final

    load_dotenv()

    BUCKET_URL: Final[str] = os.environ["NGED_S3_BUCKET_URL"]
    BUCKET: Final[str] = os.environ["NGED_S3_BUCKET"]
    ACCESS_KEY: Final[str] = os.environ["NGED_S3_BUCKET_ACCESS_KEY"]
    SECRET_KEY: Final[str] = os.environ["NGED_S3_BUCKET_SECRET"]
    REGION: Final[str] = os.environ["NGED_S3_BUCKET_REGION"]


@app.cell(disabled=True)
def _():
    http_store = obstore.store.HTTPStore(url=BUCKET_URL)
    list(http_store.list())
    return


@app.cell(disabled=True)
def _():
    s3_store = obstore.store.S3Store(
        bucket=BUCKET,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        region=REGION,
    )
    list(s3_store.list(prefix="outbound"))
    return


@app.cell
def _():
    s3_from_url = obstore.store.S3Store.from_url(
        url=BUCKET_URL,
        config={
            "aws_access_key_id": ACCESS_KEY,
            "aws_secret_access_key": SECRET_KEY,
            "aws_region": REGION,
        },
    )
    s3_from_url
    return (s3_from_url,)


@app.cell
def _(s3_from_url):
    list(s3_from_url.list("outbound"))
    return


@app.cell
def _():
    s3_from_bucket_and_config = obstore.store.S3Store(
        BUCKET,
        config={
            "aws_access_key_id": ACCESS_KEY,
            "aws_secret_access_key": SECRET_KEY,
            "aws_region": REGION,
            "aws_endpoint": BUCKET_URL,  # Explicitly set the HTTPS endpoint
            "aws_allow_http": "false",  # Enforce HTTPS (default is usually false for http)
        },
    )
    s3_from_bucket_and_config
    return (s3_from_bucket_and_config,)


@app.cell
def _(s3_from_bucket_and_config):
    list(s3_from_bucket_and_config.list("outbound"))
    return


if __name__ == "__main__":
    app.run()
