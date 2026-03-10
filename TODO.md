- [x] Before using `replace(tzinfo=None)`, make sure the datetime is in UTC
- [x] Use uint8 in Delta Lake. Maybe explicitly pass the pyarrow uint8 type. Conclusion: Delta Lake
  doesn't support unsigned ints (surprisingly!). So we'll just use Parquet.
- [x] Don't fill_na on precipitation and radiation.
- [ ] Save the H3 indicies for GB as a one-off step, perhaps in Dagster?
- [ ] Customise `opencode` agents
