import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    from dotenv import load_dotenv
    import obstore
    import os

    load_dotenv()


@app.cell(disabled=True)
def _():
    http_store = obstore.store.HTTPStore(
        url="https://nged-object-storage-dev-load-forecasting-data-eu-west-2.s3.eu-west-2.amazonaws.com/outbound/",
    )
    list(http_store.list())
    return


@app.cell(disabled=True)
def _():
    s3_store = obstore.store.S3Store(
        bucket="nged-object-storage-dev-load-forecasting-data",
        access_key_id=os.environ["NGED_S3_BUCKET_ACCESS_KEY"],
        secret_access_key=os.environ["NGED_S3_BUCKET_SECRET"],
        region=os.environ["NGED_S3_BUCKET_REGION"],
    )
    list(s3_store.list(prefix="outbound"))
    return


@app.cell
def _():
    s3_from_url = obstore.store.S3Store.from_url(
        url="https://nged-object-storage-dev-load-forecasting-data-eu-west-2.s3.eu-west-2.amazonaws.com",
        config={
            "aws_access_key_id": os.environ["NGED_S3_BUCKET_ACCESS_KEY"],
            "aws_secret_access_key": os.environ["NGED_S3_BUCKET_SECRET"],
            # Optional: Specify region if not inferred correctly from URL
            "aws_region": os.environ["NGED_S3_BUCKET_REGION"],
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
        "nged-object-storage-dev-load-forecasting-data",
        config={
            "aws_access_key_id": os.environ["NGED_S3_BUCKET_ACCESS_KEY"],
            "aws_secret_access_key": os.environ["NGED_S3_BUCKET_SECRET"],
            "aws_region": os.environ["NGED_S3_BUCKET_REGION"],
            # Explicitly set the HTTPS endpoint
            "aws_endpoint": "https://nged-object-storage-dev-load-forecasting-data-eu-west-2.s3.eu-west-2.amazonaws.com",
            "aws_allow_http": "false",  # Enforce HTTPS (default is usually false for http)
        },
    )
    s3_from_bucket_and_config
    return (s3_from_bucket_and_config,)


@app.cell
def _(s3_from_bucket_and_config):
    list(s3_from_bucket_and_config.list("outbound"))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
