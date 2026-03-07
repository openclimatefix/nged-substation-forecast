import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import obstore
    from contracts.config import Settings

    settings = Settings()

    BUCKET_URL = str(settings.NGED_S3_BUCKET_URL)
    ACCESS_KEY = settings.NGED_S3_BUCKET_ACCESS_KEY.get_secret_value()
    SECRET_KEY = settings.NGED_S3_BUCKET_SECRET.get_secret_value()


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
