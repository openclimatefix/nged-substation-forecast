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
    ACCESS_KEY: Final[str] = os.environ["NGED_S3_BUCKET_ACCESS_KEY"]
    SECRET_KEY: Final[str] = os.environ["NGED_S3_BUCKET_SECRET"]


@app.cell
def _():
    s3_from_url = obstore.store.S3Store.from_url(
        url=BUCKET_URL,
        config={
            "aws_access_key_id": ACCESS_KEY,
            "aws_secret_access_key": SECRET_KEY,
        },
    )
    s3_from_url
    return (s3_from_url,)


@app.cell
def _(s3_from_url):
    list(s3_from_url.list())
    return


if __name__ == "__main__":
    app.run()
