# Production feature audit: generated tables

Control member only; descriptive counts, no forecast scores.

## Input manifest

```json
{
  "features": [
    "power_lag_24h",
    "power_lag_48h",
    "power_lag_168h",
    "power_lag_336h",
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "local_time_of_year_sin",
    "local_time_of_year_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
    "local_day_of_week",
    "local_utc_offset_minutes",
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_short_wave_radiation_flux_surface",
    "categorical_precipitation_type_surface",
    "windchill"
  ],
  "inputs": [
    {
      "split": "Training",
      "file": "train.parquet",
      "bytes": 927982832,
      "sha256": "3b6789f0b5fa9ce7b6a14caee323104f1757558e90ec2a0365773781c810c37c",
      "all_member_rows": 9762768,
      "control_rows": 9762768,
      "series": 31,
      "first_valid_time": "2024-04-01 09:30:00+00:00",
      "last_valid_time": "2025-06-30 21:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-07-01.parquet",
      "bytes": 84746081,
      "sha256": "3404a8e34fdecd269611ab757a0c95b203ed169a0ee8f5756bc946d797c22858",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-07-01 09:30:00+00:00",
      "last_valid_time": "2025-07-16 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-07-08.parquet",
      "bytes": 83066938,
      "sha256": "9989e7766c3475552a2df6b80883e9e9c708d425bfa0dee01897169388ec14a8",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-07-08 09:30:00+00:00",
      "last_valid_time": "2025-07-23 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-07-15.parquet",
      "bytes": 84431359,
      "sha256": "241dc32f5de0b092ba7d6d28849defb6c262fbfc18ba6bca1384d3a4cc2a0c13",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-07-15 09:30:00+00:00",
      "last_valid_time": "2025-07-30 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-07-22.parquet",
      "bytes": 84135224,
      "sha256": "0e343242b402cb5a1616d5c9e853b1e148d48f8c5f94a11bf32717becae1bb39",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-07-22 09:30:00+00:00",
      "last_valid_time": "2025-08-06 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-07-29.parquet",
      "bytes": 83842829,
      "sha256": "296031ade116728e6b5034453201004b5759f5e633309cb5ddb9d9e40b4c6b88",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-07-29 09:30:00+00:00",
      "last_valid_time": "2025-08-13 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-08-05.parquet",
      "bytes": 82868821,
      "sha256": "b053a1f94455f577e39c3c54842b5096f51e5799d96b6896d89d72fe64e94de6",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-08-05 09:30:00+00:00",
      "last_valid_time": "2025-08-20 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-08-12.parquet",
      "bytes": 83664894,
      "sha256": "98a13ff6b14c09283c545bddbcd3bcbd628b74cb25fe049f6d8a674a3ed9abd4",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-08-12 09:30:00+00:00",
      "last_valid_time": "2025-08-27 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-08-19.parquet",
      "bytes": 84833836,
      "sha256": "d9e736130704adb4b860204a66ddb12ae9fe56fd6ab7abf870c78a1684b547b2",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-08-19 09:30:00+00:00",
      "last_valid_time": "2025-09-03 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-08-26.parquet",
      "bytes": 84107216,
      "sha256": "e227244750dcccff216615eee197efce17faacf0cbd2effbae73013b06fc7e51",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-08-26 09:30:00+00:00",
      "last_valid_time": "2025-09-10 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-09-02.parquet",
      "bytes": 84453312,
      "sha256": "286d6e5d84fcf2410e5fd971d51a905d8bfde6b00b747f89f1b62ed71f34de12",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-09-02 09:30:00+00:00",
      "last_valid_time": "2025-09-17 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-09-09.parquet",
      "bytes": 84522307,
      "sha256": "259d0035fa681106b00313e03ed7b8d9425824039f924940d141ff45e7caada3",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-09-09 09:30:00+00:00",
      "last_valid_time": "2025-09-24 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-09-16.parquet",
      "bytes": 85608214,
      "sha256": "ee386608daf0bfd19170fa479ee42913bec2c1f616c2cbddf01ce7ea4df44d11",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-09-16 09:30:00+00:00",
      "last_valid_time": "2025-10-01 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-09-23.parquet",
      "bytes": 83942817,
      "sha256": "ecb1e0f12f2b605fc0024b65aa3c4c20a8c9f845c766d94a1c301a1db58ee8e9",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-09-23 09:30:00+00:00",
      "last_valid_time": "2025-10-08 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-09-30.parquet",
      "bytes": 84516723,
      "sha256": "f8912be046145d792f3525f999221e8c605bab8156df15bd3d0f96355e86d73d",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-09-30 09:30:00+00:00",
      "last_valid_time": "2025-10-15 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-10-07.parquet",
      "bytes": 84531435,
      "sha256": "7640afdded2bbd80063e4989f7c3b5a210e7c86b99946fe1b2020bafa37b1af6",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-10-07 09:30:00+00:00",
      "last_valid_time": "2025-10-22 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-10-14.parquet",
      "bytes": 85657138,
      "sha256": "fa40ab33a17b77175cd9dc55fc2268fc2580d966b74e8a734c2e9befeeda4b2a",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-10-14 09:30:00+00:00",
      "last_valid_time": "2025-10-29 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-10-21.parquet",
      "bytes": 85114741,
      "sha256": "dd0a6ac42a6b3b881e337d818380005cca52188ca63e782758d7ea90b12e60b4",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-10-21 09:30:00+00:00",
      "last_valid_time": "2025-11-05 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-10-28.parquet",
      "bytes": 84253563,
      "sha256": "96ce92449daf3e40222fa4e3ff693c3c1dc506a44ec8a4f8063948f845e873b1",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-10-28 09:30:00+00:00",
      "last_valid_time": "2025-11-12 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-11-04.parquet",
      "bytes": 85276676,
      "sha256": "407e6067a2d4832d3b8431c1fbc5288ce00ab5401fd0e357993da1d690153599",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-11-04 09:30:00+00:00",
      "last_valid_time": "2025-11-19 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-11-11.parquet",
      "bytes": 86450215,
      "sha256": "00877612eedc062b12860836085129ebaa6aeb04e2a14f506cb51d49549a10d8",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-11-11 09:30:00+00:00",
      "last_valid_time": "2025-11-26 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-11-18.parquet",
      "bytes": 86349995,
      "sha256": "68836f4cdde6bcfd0251c752e90c42c8a1f839f9778111d0eb36c5c5aad067fc",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-11-18 09:30:00+00:00",
      "last_valid_time": "2025-12-03 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-11-25.parquet",
      "bytes": 85019187,
      "sha256": "def7729f821b014b402fba77136301bc0b987d949c4b1f2cebed596d21f68b11",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-11-25 09:30:00+00:00",
      "last_valid_time": "2025-12-10 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-12-02.parquet",
      "bytes": 84626356,
      "sha256": "889fbb071ddf8790bed7fcc051d6b742fa14622a889d48e1aa4c5c8a466ec2cb",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-12-02 09:30:00+00:00",
      "last_valid_time": "2025-12-17 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-12-09.parquet",
      "bytes": 83832276,
      "sha256": "73051984f5d6bca46be54d70ea467786053c07c6dd55b46c5d8b5a7ac6ad1198",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-12-09 09:30:00+00:00",
      "last_valid_time": "2025-12-24 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-12-16.parquet",
      "bytes": 85737241,
      "sha256": "6aa45043c427c7246144f1d13d28a6cd265de9074e4dbbda3e73b06ac4a9e325",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-12-16 09:30:00+00:00",
      "last_valid_time": "2025-12-31 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-12-23.parquet",
      "bytes": 84606731,
      "sha256": "7443edc204b481172af53d4197c64981d3397f8100db54f415301951b86341f6",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-12-23 09:30:00+00:00",
      "last_valid_time": "2026-01-07 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2025-12-30.parquet",
      "bytes": 85823559,
      "sha256": "d2cd36c4a1262e18a438ed4a8aec5b1199fd8b086130f07c14fd05b67a6501b4",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2025-12-30 09:30:00+00:00",
      "last_valid_time": "2026-01-14 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-01-06.parquet",
      "bytes": 87276228,
      "sha256": "a06b24e7c07e99f9afaa56eda4cfa40fd5e5b761106435535ff19bea8985814d",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-01-06 09:30:00+00:00",
      "last_valid_time": "2026-01-21 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-01-13.parquet",
      "bytes": 86166843,
      "sha256": "c5407d621633f9f82b26ed48403842509c8ea5bc00afd9a9c6000e3eb6ba1de5",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-01-13 09:30:00+00:00",
      "last_valid_time": "2026-01-28 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-01-20.parquet",
      "bytes": 86921233,
      "sha256": "a424a26f572593ecb83d900c9af33890f63798883e0072438699a40a07230e7f",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-01-20 09:30:00+00:00",
      "last_valid_time": "2026-02-04 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-01-27.parquet",
      "bytes": 86458245,
      "sha256": "24f7d38ab76d596cdeae05a4149db600b68173d36e9b0a6ee87cf915b88508e8",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-01-27 09:30:00+00:00",
      "last_valid_time": "2026-02-11 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-02-03.parquet",
      "bytes": 86743380,
      "sha256": "72ba31ce5491e30db20d970c75d4fd11fb36ba55609e2ac0af2891db8986f3fb",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-02-03 09:30:00+00:00",
      "last_valid_time": "2026-02-18 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-02-10.parquet",
      "bytes": 87438829,
      "sha256": "e1d0ea306168f710a0ca4fbe51fd4f2932bafb3c6aa4b2334de5d1ab0bfad3c9",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-02-10 09:30:00+00:00",
      "last_valid_time": "2026-02-25 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-02-17.parquet",
      "bytes": 86915967,
      "sha256": "b5c8e55e2ea3c9e79828040c165040ca9cab37722cde125f285fc073b0c80fdb",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-02-17 09:30:00+00:00",
      "last_valid_time": "2026-03-04 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-02-24.parquet",
      "bytes": 85674590,
      "sha256": "dff72e0aeeca62fbdf74adb92b75574625156b6d693456208218750a13a2bb05",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-02-24 09:30:00+00:00",
      "last_valid_time": "2026-03-11 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-03-03.parquet",
      "bytes": 86305190,
      "sha256": "b02eb31ee675f8d94b5bc4227152776ed0fddb9e0baaa6a6fc84b2cfc348d2c9",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-03-03 09:30:00+00:00",
      "last_valid_time": "2026-03-18 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-03-10.parquet",
      "bytes": 86781146,
      "sha256": "eaf50005164ca771caf585502759640f88324ad475ba8b6190bc80cf8c339100",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-03-10 09:30:00+00:00",
      "last_valid_time": "2026-03-25 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-03-17.parquet",
      "bytes": 87740472,
      "sha256": "76c6cc3ae152044ede6718d8a68732401cfff97aad18dc1cc26d87ea63abef81",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-03-17 09:30:00+00:00",
      "last_valid_time": "2026-04-01 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-03-24.parquet",
      "bytes": 87916599,
      "sha256": "c5390255679ae333a933b248ca4233ea2ec28ecd39427604520ef395a1232d8e",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-03-24 09:30:00+00:00",
      "last_valid_time": "2026-04-08 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-03-31.parquet",
      "bytes": 87500811,
      "sha256": "f5039f3417f9733b26dc5dfc5f753a7878fa1f480343e8859f4d9395313097ac",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-03-31 09:30:00+00:00",
      "last_valid_time": "2026-04-15 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-04-07.parquet",
      "bytes": 87094867,
      "sha256": "2b505dbca900ea21d91bad049e57efbcb1f8f5472bcb6e54ee5dd11cd6e9beb2",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-04-07 09:30:00+00:00",
      "last_valid_time": "2026-04-22 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-04-14.parquet",
      "bytes": 87576505,
      "sha256": "60927d2486df66befe190ad8b83121bdcb7d1e2c7d1dfd08dead401268a7de77",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-04-14 09:30:00+00:00",
      "last_valid_time": "2026-04-29 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-04-21.parquet",
      "bytes": 86448007,
      "sha256": "270215a3c763d4580939d3f870ef9dc8a232916aa0f5bfef142580811cadce62",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-04-21 09:30:00+00:00",
      "last_valid_time": "2026-05-06 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-04-28.parquet",
      "bytes": 87138038,
      "sha256": "0053bd149762dbe827ec6d3ceaeb4ff2405b47dd2eef373eaad055cc4811883a",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-04-28 09:30:00+00:00",
      "last_valid_time": "2026-05-13 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-05-05.parquet",
      "bytes": 87980215,
      "sha256": "1b9f159d3be0e51b1198bf3440bf2866aae7e927f98cd8cfdb6fea954ff42feb",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-05-05 09:30:00+00:00",
      "last_valid_time": "2026-05-20 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-05-12.parquet",
      "bytes": 87906377,
      "sha256": "88f2bb4c8c64e3a6389169599618a016b3b5bc190774784d9c50589274d2b361",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-05-12 09:30:00+00:00",
      "last_valid_time": "2026-05-27 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-05-19.parquet",
      "bytes": 85127391,
      "sha256": "b9287f14f35e68b5f506c3a023aff86e5bd74074d84c55c1fffa4dfe730c977d",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-05-19 09:30:00+00:00",
      "last_valid_time": "2026-06-03 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-05-26.parquet",
      "bytes": 85424418,
      "sha256": "acf631f7418a79e81893c6edcf7c457c9b71c91614d1f969105052f7baf2b73a",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-05-26 09:30:00+00:00",
      "last_valid_time": "2026-06-10 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-06-02.parquet",
      "bytes": 85499057,
      "sha256": "a4126c9b97d8dea23501086c9f1c5a77eb04de0b34895b8a0d701349fc7cff08",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-06-02 09:30:00+00:00",
      "last_valid_time": "2026-06-17 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-06-09.parquet",
      "bytes": 85179521,
      "sha256": "52ccfe6d83f8ce986593c06ec8e2689b1921c23f7408a245ff9c6f9c252be5a3",
      "all_member_rows": 1109862,
      "control_rows": 21762,
      "series": 31,
      "first_valid_time": "2026-06-09 09:30:00+00:00",
      "last_valid_time": "2026-06-24 00:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-06-16.parquet",
      "bytes": 81905391,
      "sha256": "de09189f02ca6675fa1228efbccf9da3c2921495f2ee4c48fd30880b7e231b17",
      "all_member_rows": 1090890,
      "control_rows": 21390,
      "series": 31,
      "first_valid_time": "2026-06-16 09:30:00+00:00",
      "last_valid_time": "2026-06-30 18:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-06-23.parquet",
      "bytes": 38129343,
      "sha256": "5c74bc52028770025639c396c0e85002bf2aaf6377923b724b695b6cb4898ec0",
      "all_member_rows": 559674,
      "control_rows": 10974,
      "series": 31,
      "first_valid_time": "2026-06-23 09:30:00+00:00",
      "last_valid_time": "2026-06-30 18:00:00+00:00"
    },
    {
      "split": "Validation",
      "file": "init=2026-06-30.parquet",
      "bytes": 1969374,
      "sha256": "bbc20d61f7c3b4036dd179505b37b431e066bbc2df37b285564ec2827d8eff90",
      "all_member_rows": 37944,
      "control_rows": 744,
      "series": 31,
      "first_valid_time": "2026-06-30 09:30:00+00:00",
      "last_valid_time": "2026-06-30 21:00:00+00:00"
    }
  ],
  "production_source_sha256": {
    "_nwp.py": "391dd1332cb2621f7c2352fd7ecb6ceb74a11034df35c4c098e88d266ce4df34",
    "_lags.py": "72ca38b6e8b34cde2222763afafe0a43fa138e8c4c8f0848018870fdb7717cfd",
    "tabular_feature_engineer.py": "6dad1c28c661e169d75a3c98a111ac3fb3043bb7e998d70383153cb586f74454",
    "solar.py": "5527ea74ab2722f7b44b244f6f441ee03bb50575ef88ecf270db6230b0aaaa71",
    "_parsed_features.py": "a15bd0103fff885e6d6e912646a8a858223bd2e8d4f4bb3301001246249fe6fa"
  },
  "polars_version": "2.0.0",
  "script_sha256": "bb737b973ce7a8701a75dc6c22e9ebb1a498d18f07ab6848db9f63a0d768234b",
  "metadata_sha256": "6a81b3f43bfa8eb1aac12fb7d0b2807ef51b55bc1082175dea667738d132d7cf",
  "model_config_sha256": "5b9bd6078453ed578b3f69da3fc08acda529c251a19fb71e2aa0934c89fd4178",
  "power_delta_version": 64,
  "power_source": "/mnt/data/nged-substation-forecast/data/NGED/cleaned_power_time_series.delta",
  "power_cutoff": "2026-07-01",
  "scope": "Forward fold, member 0 only. Current source probes do not establish historical cache provenance."
}
```

## radiation_probe

| hour | step_hours | curve | irradiance |
| --- | --- | --- | --- |
| 0.5 | 3 | truth | 0 |
| 1 | 3 | truth | 0 |
| 1.5 | 3 | truth | 0 |
| 2 | 3 | truth | 0 |
| 2.5 | 3 | truth | 0 |
| 3 | 3 | truth | 0 |
| 3.5 | 3 | truth | 0 |
| 4 | 3 | truth | 0 |
| 4.5 | 3 | truth | 0 |
| 5 | 3 | truth | 0 |
| 5.5 | 3 | truth | 0 |
| 6 | 3 | truth | 0 |
| 6.5 | 3 | truth | 52.322503 |
| 7 | 3 | truth | 156.07226 |
| 7.5 | 3 | truth | 257.15157 |
| 8 | 3 | truth | 353.83095 |
| 8.5 | 3 | truth | 444.45619 |
| 9 | 3 | truth | 527.47665 |
| 9.5 | 3 | truth | 601.47185 |
| 10 | 3 | truth | 665.17569 |
| 10.5 | 3 | truth | 717.49819 |
| 11 | 3 | truth | 757.5441 |
| 11.5 | 3 | truth | 784.62822 |
| 12 | 3 | truth | 798.28714 |
| 12.5 | 3 | truth | 798.28714 |
| 13 | 3 | truth | 784.62822 |
| 13.5 | 3 | truth | 757.5441 |
| 14 | 3 | truth | 717.49819 |
| 14.5 | 3 | truth | 665.17569 |
| 15 | 3 | truth | 601.47185 |
| 15.5 | 3 | truth | 527.47665 |
| 16 | 3 | truth | 444.45619 |
| 16.5 | 3 | truth | 353.83095 |
| 17 | 3 | truth | 257.15157 |
| 17.5 | 3 | truth | 156.07226 |
| 18 | 3 | truth | 52.322503 |
| 18.5 | 3 | truth | 0 |
| 19 | 3 | truth | 0 |
| 19.5 | 3 | truth | 0 |
| 20 | 3 | truth | 0 |
| 20.5 | 3 | truth | 0 |
| 21 | 3 | truth | 0 |
| 21.5 | 3 | truth | 0 |
| 22 | 3 | truth | 0 |
| 22.5 | 3 | truth | 0 |
| 23 | 3 | truth | 0 |
| 23.5 | 3 | truth | 0 |
| 24 | 3 | truth | 0 |
| 0.5 | 3 | production | 0 |
| 1 | 3 | production | 0 |
| 1.5 | 3 | production | 0 |
| 2 | 3 | production | 0 |
| 2.5 | 3 | production | 0 |
| 3 | 3 | production | 0 |
| 3.5 | 3 | production | 0 |
| 4 | 3 | production | 0 |
| 4.5 | 3 | production | 0 |
| 5 | 3 | production | 0 |
| 5.5 | 3 | production | 0 |
| 6 | 3 | production | 0 |
| 6.5 | 3 | production | 49.758615 |
| 7 | 3 | production | 99.517229 |
| 7.5 | 3 | production | 149.27584 |
| 8 | 3 | production | 199.03446 |
| 8.5 | 3 | production | 248.79307 |
| 9 | 3 | production | 298.55169 |
| 9.5 | 3 | production | 368.92099 |
| 10 | 3 | production | 439.2903 |
| 10.5 | 3 | production | 509.65961 |
| 11 | 3 | production | 580.02892 |
| 11.5 | 3 | production | 650.39823 |
| 12 | 3 | production | 720.76753 |
| 12.5 | 3 | production | 720.76753 |
| 13 | 3 | production | 720.76753 |
| 13.5 | 3 | production | 720.76753 |
| 14 | 3 | production | 720.76753 |
| 14.5 | 3 | production | 720.76753 |
| 15 | 3 | production | 720.76753 |
| 15.5 | 3 | production | 650.39823 |
| 16 | 3 | production | 580.02892 |
| 16.5 | 3 | production | 509.65961 |
| 17 | 3 | production | 439.2903 |
| 17.5 | 3 | production | 368.92099 |
| 18 | 3 | production | 298.55169 |
| 18.5 | 3 | production | 248.79307 |
| 19 | 3 | production | 199.03446 |
| 19.5 | 3 | production | 149.27584 |
| 20 | 3 | production | 99.517229 |
| 20.5 | 3 | production | 49.758615 |
| 21 | 3 | production | 0 |
| 21.5 | 3 | production | 0 |
| 22 | 3 | production | 0 |
| 22.5 | 3 | production | 0 |
| 23 | 3 | production | 0 |
| 23.5 | 3 | production | 0 |
| 24 | 3 | production | 0 |
| 0.5 | 6 | truth | 0 |
| 1 | 6 | truth | 0 |
| 1.5 | 6 | truth | 0 |
| 2 | 6 | truth | 0 |
| 2.5 | 6 | truth | 0 |
| 3 | 6 | truth | 0 |
| 3.5 | 6 | truth | 0 |
| 4 | 6 | truth | 0 |
| 4.5 | 6 | truth | 0 |
| 5 | 6 | truth | 0 |
| 5.5 | 6 | truth | 0 |
| 6 | 6 | truth | 0 |
| 6.5 | 6 | truth | 52.322503 |
| 7 | 6 | truth | 156.07226 |
| 7.5 | 6 | truth | 257.15157 |
| 8 | 6 | truth | 353.83095 |
| 8.5 | 6 | truth | 444.45619 |
| 9 | 6 | truth | 527.47665 |
| 9.5 | 6 | truth | 601.47185 |
| 10 | 6 | truth | 665.17569 |
| 10.5 | 6 | truth | 717.49819 |
| 11 | 6 | truth | 757.5441 |
| 11.5 | 6 | truth | 784.62822 |
| 12 | 6 | truth | 798.28714 |
| 12.5 | 6 | truth | 798.28714 |
| 13 | 6 | truth | 784.62822 |
| 13.5 | 6 | truth | 757.5441 |
| 14 | 6 | truth | 717.49819 |
| 14.5 | 6 | truth | 665.17569 |
| 15 | 6 | truth | 601.47185 |
| 15.5 | 6 | truth | 527.47665 |
| 16 | 6 | truth | 444.45619 |
| 16.5 | 6 | truth | 353.83095 |
| 17 | 6 | truth | 257.15157 |
| 17.5 | 6 | truth | 156.07226 |
| 18 | 6 | truth | 52.322503 |
| 18.5 | 6 | truth | 0 |
| 19 | 6 | truth | 0 |
| 19.5 | 6 | truth | 0 |
| 20 | 6 | truth | 0 |
| 20.5 | 6 | truth | 0 |
| 21 | 6 | truth | 0 |
| 21.5 | 6 | truth | 0 |
| 22 | 6 | truth | 0 |
| 22.5 | 6 | truth | 0 |
| 23 | 6 | truth | 0 |
| 23.5 | 6 | truth | 0 |
| 24 | 6 | truth | 0 |
| 0.5 | 6 | production | 0 |
| 1 | 6 | production | 0 |
| 1.5 | 6 | production | 0 |
| 2 | 6 | production | 0 |
| 2.5 | 6 | production | 0 |
| 3 | 6 | production | 0 |
| 3.5 | 6 | production | 0 |
| 4 | 6 | production | 0 |
| 4.5 | 6 | production | 0 |
| 5 | 6 | production | 0 |
| 5.5 | 6 | production | 0 |
| 6 | 6 | production | 0 |
| 6.5 | 6 | production | 42.471634 |
| 7 | 6 | production | 84.943268 |
| 7.5 | 6 | production | 127.4149 |
| 8 | 6 | production | 169.88654 |
| 8.5 | 6 | production | 212.35817 |
| 9 | 6 | production | 254.8298 |
| 9.5 | 6 | production | 297.30144 |
| 10 | 6 | production | 339.77307 |
| 10.5 | 6 | production | 382.24471 |
| 11 | 6 | production | 424.71634 |
| 11.5 | 6 | production | 467.18798 |
| 12 | 6 | production | 509.65961 |
| 12.5 | 6 | production | 509.65961 |
| 13 | 6 | production | 509.65961 |
| 13.5 | 6 | production | 509.65961 |
| 14 | 6 | production | 509.65961 |
| 14.5 | 6 | production | 509.65961 |
| 15 | 6 | production | 509.65961 |
| 15.5 | 6 | production | 509.65961 |
| 16 | 6 | production | 509.65961 |
| 16.5 | 6 | production | 509.65961 |
| 17 | 6 | production | 509.65961 |
| 17.5 | 6 | production | 509.65961 |
| 18 | 6 | production | 509.65961 |
| 18.5 | 6 | production | 467.18798 |
| 19 | 6 | production | 424.71634 |
| 19.5 | 6 | production | 382.24471 |
| 20 | 6 | production | 339.77307 |
| 20.5 | 6 | production | 297.30144 |
| 21 | 6 | production | 254.8298 |
| 21.5 | 6 | production | 212.35817 |
| 22 | 6 | production | 169.88654 |
| 22.5 | 6 | production | 127.4149 |
| 23 | 6 | production | 84.943268 |
| 23.5 | 6 | production | 42.471634 |
| 24 | 6 | production | 0 |

## radiation_probe_metrics

| step_hours | centroid_delay_hours | truth_energy_kwh_m2 | reconstructed_energy_kwh_m2 | dark_positive_rows | eligible_rows | leading_null_rows |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 1.25 | 6.1159153 | 6.1159153 | 5 | 48 | 6 |
| 6 | 2.75 | 6.1159153 | 6.1159153 | 11 | 48 | 12 |

## direction_probe

| hour | production | shortest_arc | case | circular_discrepancy |
| --- | --- | --- | --- | --- |
| 0 | 350 | 350 | Crossing north | 0 |
| 0.5 | 293.33333 | 353.33333 | Crossing north | 60 |
| 1 | 236.66667 | 356.66667 | Crossing north | 120 |
| 1.5 | 180 | 0 | Crossing north | 180 |
| 2 | 123.33333 | 3.3333333 | Crossing north | 120 |
| 2.5 | 66.666667 | 6.6666667 | Crossing north | 60 |
| 3 | 10 | 10 | Crossing north | 0 |
| 0 | 100 | 100 | No wrap control | 0 |
| 0.5 | 103.33333 | 103.33333 | No wrap control | 0 |
| 1 | 106.66667 | 106.66667 | No wrap control | 0 |
| 1.5 | 110 | 110 | No wrap control | 0 |
| 2 | 113.33333 | 113.33333 | No wrap control | 0 |
| 2.5 | 116.66667 | 116.66667 | No wrap control | 0 |
| 3 | 120 | 120 | No wrap control | 0 |

## weekday_probe

| weekday | ordinal | sine | cosine |
| --- | --- | --- | --- |
| Monday | 0 | 0.7818315 | 0.6234898 |
| Tuesday | 1 | 0.9749279 | -0.22252093 |
| Wednesday | 2 | 0.43388373 | -0.90096885 |
| Thursday | 3 | -0.43388373 | -0.90096885 |
| Friday | 4 | -0.9749279 | -0.22252093 |
| Saturday | 5 | -0.7818315 | 0.6234898 |
| Sunday | 6 | -2.4492937e-16 | 1 |

## lag_boundary_probe

| lead_hours | value |
| --- | --- |
| 335.5 | 1 |
| 336 | None |
| 336.5 | None |

## ranges

| feature | split | rows | missing | nonfinite | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| categorical_precipitation_type_surface | Training | 9762768 | 4918212 | 0 | 0 | 12 |
| categorical_precipitation_type_surface | Validation | 1121208 | 0 | 0 | 0 | 12 |
| dew_point_temperature_2m | Training | 9762768 | 0 | 0 | -15.597656 | 21.421875 |
| dew_point_temperature_2m | Validation | 1121208 | 0 | 0 | -9.4804688 | 22.152344 |
| downward_short_wave_radiation_flux_surface | Training | 9762768 | 0 | 0 | 0 | 816 |
| downward_short_wave_radiation_flux_surface | Validation | 1121208 | 0 | 0 | 0 | 808 |
| geopotential_height_500hpa | Training | 9762768 | 0 | 0 | 5032 | 5952 |
| geopotential_height_500hpa | Validation | 1121208 | 0 | 0 | 5166 | 5934 |
| local_day_of_week | Training | 9762768 | 0 | 0 | 0 | 6 |
| local_day_of_week | Validation | 1121208 | 0 | 0 | 0 | 6 |
| local_day_of_week_cos | Training | 9762768 | 0 | 0 | -0.90096885 | 1 |
| local_day_of_week_cos | Validation | 1121208 | 0 | 0 | -0.90096885 | 1 |
| local_day_of_week_sin | Training | 9762768 | 0 | 0 | -0.9749279 | 0.9749279 |
| local_day_of_week_sin | Validation | 1121208 | 0 | 0 | -0.9749279 | 0.9749279 |
| local_time_of_day_cos | Training | 9762768 | 0 | 0 | -1 | 1 |
| local_time_of_day_cos | Validation | 1121208 | 0 | 0 | -1 | 1 |
| local_time_of_day_sin | Training | 9762768 | 0 | 0 | -1 | 1 |
| local_time_of_day_sin | Validation | 1121208 | 0 | 0 | -1 | 1 |
| local_time_of_year_cos | Training | 9762768 | 0 | 0 | -1 | 1 |
| local_time_of_year_cos | Validation | 1121208 | 0 | 0 | -1 | 0.99985266 |
| local_time_of_year_sin | Training | 9762768 | 0 | 0 | -0.99996316 | 0.99996316 |
| local_time_of_year_sin | Validation | 1121208 | 0 | 0 | -0.99996316 | 0.99996316 |
| local_utc_offset_minutes | Training | 9762768 | 0 | 0 | 0 | 60 |
| local_utc_offset_minutes | Validation | 1121208 | 0 | 0 | 0 | 60 |
| power_lag_168h | Training | 9762768 | 5334287 | 0 | None | None |
| power_lag_168h | Validation | 1121208 | 604278 | 0 | None | None |
| power_lag_24h | Training | 9762768 | 9136937 | 0 | None | None |
| power_lag_24h | Validation | 1121208 | 1047396 | 0 | None | None |
| power_lag_336h | Training | 9762768 | 967647 | 0 | None | None |
| power_lag_336h | Validation | 1121208 | 94864 | 0 | None | None |
| power_lag_48h | Training | 9762768 | 8499188 | 0 | None | None |
| power_lag_48h | Validation | 1121208 | 972480 | 0 | None | None |
| pressure_reduced_to_mean_sea_level | Training | 9762768 | 0 | 0 | 96208 | 104960 |
| pressure_reduced_to_mean_sea_level | Validation | 1121208 | 0 | 0 | 97536 | 104064 |
| pressure_surface | Training | 9762768 | 0 | 0 | 95520 | 104928 |
| pressure_surface | Validation | 1121208 | 0 | 0 | 96768 | 103968 |
| temperature_2m | Training | 9762768 | 0 | 0 | -9.1484375 | 32.75 |
| temperature_2m | Validation | 1121208 | 0 | 0 | -4.8798828 | 32.617188 |
| wind_direction_100m | Training | 9762768 | 0 | 0 | 0.0010681152 | 360 |
| wind_direction_100m | Validation | 1121208 | 0 | 0 | 0.037017822 | 360 |
| wind_direction_10m | Training | 9762768 | 0 | 0 | 0.0013275146 | 360 |
| wind_direction_10m | Validation | 1121208 | 0 | 0 | 0.037063599 | 360 |
| wind_speed_100m | Training | 9762768 | 0 | 0 | 0.012527466 | 27.859375 |
| wind_speed_100m | Validation | 1121208 | 0 | 0 | 0.077804565 | 22.082031 |
| wind_speed_10m | Training | 9762768 | 0 | 0 | 0.012460709 | 20.8125 |
| wind_speed_10m | Validation | 1121208 | 0 | 0 | 0.010866165 | 15.892578 |
| windchill | Training | 9762768 | 0 | 0 | -13.447577 | 35.992989 |
| windchill | Validation | 1121208 | 0 | 0 | -9.0952244 | 35.735062 |

## precip_runs

| nwp_init_time | split | rows | missing | series |
| --- | --- | --- | --- | --- |
| 2024-04-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-04-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-05-31 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-06-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-07-31 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-08-31 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-09-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-13 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-14 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-15 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-16 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-17 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-18 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-19 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-20 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-21 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-22 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-23 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-24 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-25 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-26 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-27 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-28 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-29 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-30 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-10-31 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-01 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-02 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-03 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-04 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-05 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-06 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-07 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-08 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-09 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-10 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-11 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-12 00:00:00+00:00 | Training | 21762 | 21762 | 31 |
| 2024-11-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-29 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-11-30 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-29 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-30 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2024-12-31 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-29 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-30 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-01-31 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-02-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-29 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-30 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-03-31 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-29 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-04-30 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-16 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-17 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-18 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-19 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-20 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-21 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-22 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-23 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-24 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-25 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-26 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-27 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-28 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-29 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-30 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-05-31 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-01 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-02 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-03 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-04 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-05 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-06 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-07 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-08 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-09 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-10 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-11 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-12 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-13 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-14 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-15 00:00:00+00:00 | Training | 21762 | 0 | 31 |
| 2025-06-16 00:00:00+00:00 | Training | 21390 | 0 | 31 |
| 2025-06-17 00:00:00+00:00 | Training | 19902 | 0 | 31 |
| 2025-06-18 00:00:00+00:00 | Training | 18414 | 0 | 31 |
| 2025-06-19 00:00:00+00:00 | Training | 16926 | 0 | 31 |
| 2025-06-20 00:00:00+00:00 | Training | 15438 | 0 | 31 |
| 2025-06-21 00:00:00+00:00 | Training | 13950 | 0 | 31 |
| 2025-06-22 00:00:00+00:00 | Training | 12462 | 0 | 31 |
| 2025-06-23 00:00:00+00:00 | Training | 10974 | 0 | 31 |
| 2025-06-24 00:00:00+00:00 | Training | 9486 | 0 | 31 |
| 2025-06-25 00:00:00+00:00 | Training | 8184 | 0 | 31 |
| 2025-06-26 00:00:00+00:00 | Training | 6696 | 0 | 31 |
| 2025-06-27 00:00:00+00:00 | Training | 5208 | 0 | 31 |
| 2025-06-28 00:00:00+00:00 | Training | 3720 | 0 | 31 |
| 2025-06-29 00:00:00+00:00 | Training | 2232 | 0 | 31 |
| 2025-06-30 00:00:00+00:00 | Training | 744 | 0 | 31 |
| 2025-07-01 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-07-08 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-07-15 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-07-22 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-07-29 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-08-05 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-08-12 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-08-19 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-08-26 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-09-02 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-09-09 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-09-16 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-09-23 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-09-30 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-10-07 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-10-14 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-10-21 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-10-28 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-11-04 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-11-11 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-11-18 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-11-25 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-12-02 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-12-09 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-12-16 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-12-23 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2025-12-30 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-01-06 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-01-13 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-01-20 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-01-27 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-02-03 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-02-10 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-02-17 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-02-24 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-03-03 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-03-10 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-03-17 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-03-24 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-03-31 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-04-07 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-04-14 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-04-21 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-04-28 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-05-05 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-05-12 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-05-19 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-05-26 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-06-02 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-06-09 00:00:00+00:00 | Validation | 21762 | 0 | 31 |
| 2026-06-16 00:00:00+00:00 | Validation | 21390 | 0 | 31 |
| 2026-06-23 00:00:00+00:00 | Validation | 10974 | 0 | 31 |
| 2026-06-30 00:00:00+00:00 | Validation | 744 | 0 | 31 |

## precip_population

| band | split | eligible | affected | affected_series | percent |
| --- | --- | --- | --- | --- | --- |
| 0–72 h | Training | 2014845 | 1001858 | 31 | 49.723825 |
| 0–72 h | Validation | 231260 | 0 | 0 | 0 |
| 144–264 h | Training | 3328625 | 1681440 | 31 | 50.514552 |
| 144–264 h | Validation | 381517 | 0 | 0 | 0 |
| 264 h and later | Training | 2403709 | 1226050 | 31 | 51.00659 |
| 264 h and later | Validation | 276303 | 0 | 0 | 0 |
| 72–144 h | Training | 2015589 | 1008864 | 31 | 50.053061 |
| 72–144 h | Validation | 232128 | 0 | 0 | 0 |

## windchill

| band | split | eligible | affected | affected_series | percent |
| --- | --- | --- | --- | --- | --- |
| 0–72 h | Training | 2014845 | 1218331 | 31 | 60.467728 |
| 0–72 h | Validation | 231260 | 136656 | 31 | 59.091931 |
| 144–264 h | Training | 3328625 | 1967806 | 31 | 59.117684 |
| 144–264 h | Validation | 381517 | 203208 | 31 | 53.263157 |
| 264 h and later | Training | 2403709 | 1403426 | 31 | 58.385853 |
| 264 h and later | Validation | 276303 | 152012 | 31 | 55.016413 |
| 72–144 h | Training | 2015589 | 1225751 | 31 | 60.813539 |
| 72–144 h | Validation | 232128 | 125803 | 31 | 54.19553 |

## formula

| split | max_abs_formula_difference | outside_rows | outside_above_air |
| --- | --- | --- | --- |
| Training | 0 | 5815314 | 1115084 |
| Validation | 0 | 617679 | 148593 |

## direction

| band | split | height_m | eligible | affected | affected_series | percent |
| --- | --- | --- | --- | --- | --- | --- |
| 0–72 h | Training | 10 | 1620060 | 60475 | 31 | 3.7328864 |
| 0–72 h | Training | 100 | 1620060 | 54945 | 31 | 3.3915411 |
| 0–72 h | Validation | 10 | 185845 | 6300 | 31 | 3.3899217 |
| 0–72 h | Validation | 100 | 185845 | 6010 | 31 | 3.2338777 |
| 144–264 h | Training | 10 | 3051175 | 211489 | 31 | 6.9313953 |
| 144–264 h | Training | 100 | 3051142 | 192933 | 31 | 6.3233045 |
| 144–264 h | Validation | 10 | 349711 | 22387 | 31 | 6.4015716 |
| 144–264 h | Validation | 100 | 349711 | 19862 | 31 | 5.6795468 |
| 264 h and later | Training | 10 | 2197652 | 153895 | 31 | 7.0027011 |
| 264 h and later | Training | 100 | 2197652 | 138634 | 31 | 6.3082781 |
| 264 h and later | Validation | 10 | 252619 | 15193 | 31 | 6.0141953 |
| 264 h and later | Validation | 100 | 252619 | 13353 | 31 | 5.2858257 |
| 72–144 h | Training | 10 | 1693530 | 72781 | 31 | 4.2975914 |
| 72–144 h | Training | 100 | 1693530 | 65832 | 31 | 3.8872651 |
| 72–144 h | Validation | 10 | 195052 | 5940 | 31 | 3.0453418 |
| 72–144 h | Validation | 100 | 195052 | 5429 | 31 | 2.7833603 |

## radiation

| band | grid | split | rows | dark_rows | dark_positive | native_dark_interval_rows | native_dark_interval_positive | affected_series |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0–72 h | Inserted half-hour | Training | 1690740 | 744272 | 316810 | 0 | 0 | 31 |
| 0–72 h | Inserted half-hour | Validation | 194060 | 90719 | 36121 | 0 | 0 | 31 |
| 0–72 h | Native interval endpoint | Training | 324105 | 156480 | 37120 | 116823 | 0 | 31 |
| 0–72 h | Native interval endpoint | Validation | 37200 | 18797 | 4102 | 14322 | 0 | 31 |
| 144–264 h | Inserted half-hour | Training | 3051175 | 1360480 | 1009383 | 0 | 0 | 31 |
| 144–264 h | Inserted half-hour | Validation | 349711 | 165722 | 115043 | 0 | 0 | 31 |
| 144–264 h | Native interval endpoint | Training | 277450 | 118820 | 59931 | 53965 | 0 | 31 |
| 144–264 h | Native interval endpoint | Validation | 31806 | 15032 | 6587 | 7750 | 0 | 31 |
| 264 h and later | Inserted half-hour | Training | 2197652 | 937376 | 727379 | 0 | 0 | 31 |
| 264 h and later | Inserted half-hour | Validation | 252619 | 114891 | 85059 | 0 | 0 | 31 |
| 264 h and later | Native interval endpoint | Training | 206057 | 89117 | 47570 | 37791 | 0 | 31 |
| 264 h and later | Native interval endpoint | Validation | 23684 | 11178 | 5271 | 5364 | 0 | 31 |
| 72–144 h | Inserted half-hour | Training | 1693530 | 753420 | 355648 | 0 | 0 | 31 |
| 72–144 h | Inserted half-hour | Validation | 195052 | 92394 | 39747 | 0 | 0 | 31 |
| 72–144 h | Native interval endpoint | Training | 322059 | 141693 | 36921 | 102315 | 0 | 31 |
| 72–144 h | Native interval endpoint | Validation | 37076 | 17116 | 3997 | 12649 | 0 | 31 |

## radiation_profile

| band | split | hour | rows | radiation_sum |
| --- | --- | --- | --- | --- |
| 0–72 h | Training | 0 | 84444 | 80.88427 |
| 0–72 h | Training | 1 | 84444 | 158.06876 |
| 0–72 h | Training | 2 | 84444 | 235.25342 |
| 0–72 h | Training | 3 | 84444 | 135152.22 |
| 0–72 h | Training | 4 | 84444 | 674586.5 |
| 0–72 h | Training | 5 | 84444 | 1214020.1 |
| 0–72 h | Training | 6 | 84444 | 2821459 |
| 0–72 h | Training | 7 | 84444 | 7632901 |
| 0–72 h | Training | 8 | 84444 | 12444356 |
| 0–72 h | Training | 9 | 70494 | 14868015 |
| 0–72 h | Training | 10 | 84630 | 23462198 |
| 0–72 h | Training | 11 | 84630 | 29391730 |
| 0–72 h | Training | 12 | 84630 | 33737904 |
| 0–72 h | Training | 13 | 84630 | 33334092 |
| 0–72 h | Training | 14 | 84630 | 32930246 |
| 0–72 h | Training | 15 | 84630 | 31172796 |
| 0–72 h | Training | 16 | 84630 | 25354464 |
| 0–72 h | Training | 17 | 84630 | 19536148 |
| 0–72 h | Training | 18 | 84630 | 14042064 |
| 0–72 h | Training | 19 | 84630 | 9520748 |
| 0–72 h | Training | 20 | 84630 | 4999424 |
| 0–72 h | Training | 21 | 84537 | 1468496.8 |
| 0–72 h | Training | 22 | 84444 | 930016 |
| 0–72 h | Training | 23 | 84444 | 398613.41 |
| 0–72 h | Validation | 0 | 9672 | 12.103111 |
| 0–72 h | Validation | 1 | 9672 | 17.822813 |
| 0–72 h | Validation | 2 | 9672 | 23.542513 |
| 0–72 h | Validation | 3 | 9672 | 13049.477 |
| 0–72 h | Validation | 4 | 9672 | 65136.055 |
| 0–72 h | Validation | 5 | 9672 | 117222.62 |
| 0–72 h | Validation | 6 | 9672 | 278789.25 |
| 0–72 h | Validation | 7 | 9672 | 768795.94 |
| 0–72 h | Validation | 8 | 9672 | 1258802.6 |
| 0–72 h | Validation | 9 | 8091 | 1498428.6 |
| 0–72 h | Validation | 10 | 9734 | 2386011.8 |
| 0–72 h | Validation | 11 | 9734 | 3001390.2 |
| 0–72 h | Validation | 12 | 9734 | 3449335.8 |
| 0–72 h | Validation | 13 | 9734 | 3394982.5 |
| 0–72 h | Validation | 14 | 9734 | 3340629.5 |
| 0–72 h | Validation | 15 | 9734 | 3152354 |
| 0–72 h | Validation | 16 | 9734 | 2562311 |
| 0–72 h | Validation | 17 | 9734 | 1972268.2 |
| 0–72 h | Validation | 18 | 9734 | 1415243.2 |
| 0–72 h | Validation | 19 | 9734 | 957271.88 |
| 0–72 h | Validation | 20 | 9734 | 499300.5 |
| 0–72 h | Validation | 21 | 9703 | 141085.09 |
| 0–72 h | Validation | 22 | 9672 | 88446.938 |
| 0–72 h | Validation | 23 | 9672 | 37911.93 |
| 144–264 h | Training | 0 | 138570 | 1343411.6 |
| 144–264 h | Training | 1 | 138570 | 1345386.8 |
| 144–264 h | Training | 2 | 138570 | 1347365.1 |
| 144–264 h | Training | 3 | 138570 | 1349338 |
| 144–264 h | Training | 4 | 138570 | 1351315.1 |
| 144–264 h | Training | 5 | 138570 | 1353289.2 |
| 144–264 h | Training | 6 | 138570 | 2998673.2 |
| 144–264 h | Training | 7 | 138570 | 9574301 |
| 144–264 h | Training | 8 | 138570 | 16149914 |
| 144–264 h | Training | 9 | 138880 | 22768678 |
| 144–264 h | Training | 10 | 138880 | 29360376 |
| 144–264 h | Training | 11 | 138880 | 35952108 |
| 144–264 h | Training | 12 | 138880 | 40819880 |
| 144–264 h | Training | 13 | 138880 | 40515944 |
| 144–264 h | Training | 14 | 138880 | 40212064 |
| 144–264 h | Training | 15 | 138880 | 39908108 |
| 144–264 h | Training | 16 | 138880 | 39604200 |
| 144–264 h | Training | 17 | 138880 | 39300236 |
| 144–264 h | Training | 18 | 138725 | 37422812 |
| 144–264 h | Training | 19 | 138570 | 31078218 |
| 144–264 h | Training | 20 | 138570 | 24818160 |
| 144–264 h | Training | 21 | 138570 | 18558118 |
| 144–264 h | Training | 22 | 138570 | 12298032 |
| 144–264 h | Training | 23 | 138570 | 6037969.5 |
| 144–264 h | Validation | 0 | 15872 | 120658.48 |
| 144–264 h | Validation | 1 | 15872 | 119453.66 |
| 144–264 h | Validation | 2 | 15872 | 118248.84 |
| 144–264 h | Validation | 3 | 15872 | 117044.02 |
| 144–264 h | Validation | 4 | 15872 | 115839.2 |
| 144–264 h | Validation | 5 | 15872 | 114634.38 |
| 144–264 h | Validation | 6 | 15872 | 279638.91 |
| 144–264 h | Validation | 7 | 15872 | 943271.5 |
| 144–264 h | Validation | 8 | 15872 | 1606904.1 |
| 144–264 h | Validation | 9 | 15934 | 2273877 |
| 144–264 h | Validation | 10 | 15934 | 2938414.2 |
| 144–264 h | Validation | 11 | 15934 | 3602951.5 |
| 144–264 h | Validation | 12 | 15934 | 4097279.5 |
| 144–264 h | Validation | 13 | 15934 | 4080980.2 |
| 144–264 h | Validation | 14 | 15934 | 4064681 |
| 144–264 h | Validation | 15 | 15934 | 4048382 |
| 144–264 h | Validation | 16 | 15934 | 4032082.5 |
| 144–264 h | Validation | 17 | 15934 | 4015783.5 |
| 144–264 h | Validation | 18 | 15903 | 3826726 |
| 144–264 h | Validation | 19 | 15872 | 3168674.8 |
| 144–264 h | Validation | 20 | 15872 | 2527050.5 |
| 144–264 h | Validation | 21 | 15872 | 1885426.4 |
| 144–264 h | Validation | 22 | 15872 | 1243802.1 |
| 144–264 h | Validation | 23 | 15872 | 602177.88 |
| 264 h and later | Training | 0 | 96069 | 937381.56 |
| 264 h and later | Training | 1 | 82398 | 807461.19 |
| 264 h and later | Training | 2 | 82398 | 813336.69 |
| 264 h and later | Training | 3 | 82398 | 819215.69 |
| 264 h and later | Training | 4 | 82398 | 825093.19 |
| 264 h and later | Training | 5 | 82398 | 830969.5 |
| 264 h and later | Training | 6 | 82398 | 1816784.5 |
| 264 h and later | Training | 7 | 82398 | 5742403 |
| 264 h and later | Training | 8 | 82398 | 9668023 |
| 264 h and later | Training | 9 | 109988 | 18150976 |
| 264 h and later | Training | 10 | 109988 | 23394818 |
| 264 h and later | Training | 11 | 109988 | 28638738 |
| 264 h and later | Training | 12 | 109988 | 32485410 |
| 264 h and later | Training | 13 | 109988 | 32140882 |
| 264 h and later | Training | 14 | 109988 | 31796240 |
| 264 h and later | Training | 15 | 109988 | 31451682 |
| 264 h and later | Training | 16 | 109988 | 31107044 |
| 264 h and later | Training | 17 | 109988 | 30762490 |
| 264 h and later | Training | 18 | 109864 | 29222072 |
| 264 h and later | Training | 19 | 109740 | 24276320 |
| 264 h and later | Training | 20 | 109740 | 19391166 |
| 264 h and later | Training | 21 | 109740 | 14505965 |
| 264 h and later | Training | 22 | 109740 | 9620779 |
| 264 h and later | Training | 23 | 109740 | 4735597.5 |
| 264 h and later | Validation | 0 | 11036 | 80608.016 |
| 264 h and later | Validation | 1 | 9486 | 68241.711 |
| 264 h and later | Validation | 2 | 9486 | 67610.867 |
| 264 h and later | Validation | 3 | 9486 | 66980.023 |
| 264 h and later | Validation | 4 | 9486 | 66349.18 |
| 264 h and later | Validation | 5 | 9486 | 65718.336 |
| 264 h and later | Validation | 6 | 9486 | 165849.59 |
| 264 h and later | Validation | 7 | 9486 | 568267.12 |
| 264 h and later | Validation | 8 | 9486 | 970684.69 |
| 264 h and later | Validation | 9 | 12648 | 1827688.6 |
| 264 h and later | Validation | 10 | 12648 | 2363147.8 |
| 264 h and later | Validation | 11 | 12648 | 2898606.8 |
| 264 h and later | Validation | 12 | 12648 | 3294580.8 |
| 264 h and later | Validation | 13 | 12648 | 3272100.2 |
| 264 h and later | Validation | 14 | 12648 | 3249619.5 |
| 264 h and later | Validation | 15 | 12648 | 3227138.8 |
| 264 h and later | Validation | 16 | 12648 | 3204658 |
| 264 h and later | Validation | 17 | 12648 | 3182177.5 |
| 264 h and later | Validation | 18 | 12617 | 3020538.2 |
| 264 h and later | Validation | 19 | 12586 | 2496224.5 |
| 264 h and later | Validation | 20 | 12586 | 1990177.4 |
| 264 h and later | Validation | 21 | 12586 | 1484130.4 |
| 264 h and later | Validation | 22 | 12586 | 978083.44 |
| 264 h and later | Validation | 23 | 12586 | 472036.44 |
| 72–144 h | Training | 0 | 83886 | 11096.426 |
| 72–144 h | Training | 1 | 83886 | 55100.109 |
| 72–144 h | Training | 2 | 83886 | 99103.797 |
| 72–144 h | Training | 3 | 83886 | 234172.95 |
| 72–144 h | Training | 4 | 83886 | 642438.12 |
| 72–144 h | Training | 5 | 83886 | 1050704.6 |
| 72–144 h | Training | 6 | 83886 | 2501019 |
| 72–144 h | Training | 7 | 83886 | 7077489 |
| 72–144 h | Training | 8 | 83886 | 11653957 |
| 72–144 h | Training | 9 | 84072 | 17617218 |
| 72–144 h | Training | 10 | 84072 | 23480646 |
| 72–144 h | Training | 11 | 84072 | 29344052 |
| 72–144 h | Training | 12 | 84072 | 33633524 |
| 72–144 h | Training | 13 | 84072 | 33201106 |
| 72–144 h | Training | 14 | 84072 | 32768772 |
| 72–144 h | Training | 15 | 84072 | 30995008 |
| 72–144 h | Training | 16 | 84072 | 25197374 |
| 72–144 h | Training | 17 | 84072 | 19399710 |
| 72–144 h | Training | 18 | 84072 | 13931158 |
| 72–144 h | Training | 19 | 84072 | 9449974 |
| 72–144 h | Training | 20 | 84072 | 4968767.5 |
| 72–144 h | Training | 21 | 83979 | 1467854.8 |
| 72–144 h | Training | 22 | 83886 | 929509.62 |
| 72–144 h | Training | 23 | 83886 | 398416.03 |
| 72–144 h | Validation | 0 | 9672 | 958.92023 |
| 72–144 h | Validation | 1 | 9672 | 4727.9902 |
| 72–144 h | Validation | 2 | 9672 | 8497.0605 |
| 72–144 h | Validation | 3 | 9672 | 20471.809 |
| 72–144 h | Validation | 4 | 9672 | 57063.594 |
| 72–144 h | Validation | 5 | 9672 | 93655.383 |
| 72–144 h | Validation | 6 | 9672 | 232500.12 |
| 72–144 h | Validation | 7 | 9672 | 678103.75 |
| 72–144 h | Validation | 8 | 9672 | 1123707.4 |
| 72–144 h | Validation | 9 | 9672 | 1742663.8 |
| 72–144 h | Validation | 10 | 9672 | 2363725 |
| 72–144 h | Validation | 11 | 9672 | 2984786.5 |
| 72–144 h | Validation | 12 | 9672 | 3439325.2 |
| 72–144 h | Validation | 13 | 9672 | 3394296.5 |
| 72–144 h | Validation | 14 | 9672 | 3349267.8 |
| 72–144 h | Validation | 15 | 9672 | 3161469.8 |
| 72–144 h | Validation | 16 | 9672 | 2545363.2 |
| 72–144 h | Validation | 17 | 9672 | 1929256.9 |
| 72–144 h | Validation | 18 | 9672 | 1357396.6 |
| 72–144 h | Validation | 19 | 9672 | 918274.94 |
| 72–144 h | Validation | 20 | 9672 | 479153.16 |
| 72–144 h | Validation | 21 | 9672 | 137328.88 |
| 72–144 h | Validation | 22 | 9672 | 87397.164 |
| 72–144 h | Validation | 23 | 9672 | 37465.441 |

## windchill_profile

| domain | split | temperature_bin | rows | excess_sum |
| --- | --- | --- | --- | --- |
| Inside formula domain | Training | -10 | 133 | -564.91309 |
| Inside formula domain | Training | -8 | 962 | -4135.4282 |
| Inside formula domain | Training | -6 | 2644 | -10449.978 |
| Inside formula domain | Training | -4 | 24226 | -90760 |
| Inside formula domain | Training | -2 | 107275 | -398673.44 |
| Inside formula domain | Training | 0 | 310495 | -1146967 |
| Inside formula domain | Training | 2 | 548440 | -1938274.1 |
| Inside formula domain | Training | 4 | 790919 | -2595469 |
| Inside formula domain | Training | 6 | 1024359 | -2974547 |
| Inside formula domain | Training | 8 | 1137497 | -2763528 |
| Inside formula domain | Training | 10 | 504 | -1198.3629 |
| Inside formula domain | Validation | -6 | 15 | -52.850239 |
| Inside formula domain | Validation | -4 | 2610 | -11274.408 |
| Inside formula domain | Validation | -2 | 10791 | -46064.664 |
| Inside formula domain | Validation | 0 | 33139 | -134649.77 |
| Inside formula domain | Validation | 2 | 64673 | -237902.42 |
| Inside formula domain | Validation | 4 | 121616 | -405452.34 |
| Inside formula domain | Validation | 6 | 132670 | -401646.5 |
| Inside formula domain | Validation | 8 | 137995 | -360690.22 |
| Inside formula domain | Validation | 10 | 20 | -42.956535 |
| Outside formula domain | Training | -8 | 76 | -108.36025 |
| Outside formula domain | Training | -6 | 1285 | -1466.6464 |
| Outside formula domain | Training | -4 | 4962 | -2810.696 |
| Outside formula domain | Training | -2 | 14271 | -8415.4404 |
| Outside formula domain | Training | 0 | 21832 | -11099.926 |
| Outside formula domain | Training | 2 | 25293 | -8008.6138 |
| Outside formula domain | Training | 4 | 30165 | -3328.0078 |
| Outside formula domain | Training | 6 | 34048 | 2134.718 |
| Outside formula domain | Training | 8 | 29456 | 7173.4565 |
| Outside formula domain | Training | 10 | 1259659 | -2414359 |
| Outside formula domain | Training | 12 | 1253168 | -1814713.1 |
| Outside formula domain | Training | 14 | 1144336 | -1097996 |
| Outside formula domain | Training | 16 | 884814 | -427899.5 |
| Outside formula domain | Training | 18 | 601727 | -18695.824 |
| Outside formula domain | Training | 20 | 316125 | 136300.34 |
| Outside formula domain | Training | 22 | 137084 | 123579.62 |
| Outside formula domain | Training | 24 | 42347 | 57906.434 |
| Outside formula domain | Training | 26 | 10047 | 18465.717 |
| Outside formula domain | Training | 28 | 3564 | 8274.1748 |
| Outside formula domain | Training | 30 | 1026 | 2756.5286 |
| Outside formula domain | Training | 32 | 29 | 89.055992 |
| Outside formula domain | Validation | -6 | 1 | -1.8772631 |
| Outside formula domain | Validation | -4 | 489 | -433.0896 |
| Outside formula domain | Validation | -2 | 501 | -499.36322 |
| Outside formula domain | Validation | 0 | 1262 | -639.01733 |
| Outside formula domain | Validation | 2 | 3071 | -1228.9534 |
| Outside formula domain | Validation | 4 | 2672 | -670.01654 |
| Outside formula domain | Validation | 6 | 2519 | 63.887657 |
| Outside formula domain | Validation | 8 | 2810 | 439.73557 |
| Outside formula domain | Validation | 10 | 120115 | -235405.67 |
| Outside formula domain | Validation | 12 | 115862 | -163733.58 |
| Outside formula domain | Validation | 14 | 122086 | -116134.13 |
| Outside formula domain | Validation | 16 | 103699 | -45747.941 |
| Outside formula domain | Validation | 18 | 72026 | 2504.5474 |
| Outside formula domain | Validation | 20 | 39547 | 20485.824 |
| Outside formula domain | Validation | 22 | 17319 | 17708.75 |
| Outside formula domain | Validation | 24 | 8557 | 12475.422 |
| Outside formula domain | Validation | 26 | 3613 | 6786.2939 |
| Outside formula domain | Validation | 28 | 1147 | 2597.8887 |
| Outside formula domain | Validation | 30 | 360 | 970.41028 |
| Outside formula domain | Validation | 32 | 23 | 69.373222 |

## temperature

| split | days | affected_days | mean_range_3h | mean_range_6h | mean_range_loss |
| --- | --- | --- | --- | --- | --- |
| Training | 70215 | 61503 | 5.8504181 | 4.9766259 | 0.87379181 |
| Validation | 8060 | 6828 | 5.4278293 | 4.5653586 | 0.86247072 |

## lag_missing

| band | split | eligible | affected | affected_series | percent |
| --- | --- | --- | --- | --- | --- |
| 0–72 h | Training | 2014845 | 117918 | 19 | 5.8524601 |
| 0–72 h | Validation | 231260 | 13553 | 31 | 5.8605033 |
| 144–264 h | Training | 3328625 | 193440 | 19 | 5.8114086 |
| 144–264 h | Validation | 381517 | 18306 | 31 | 4.7982134 |
| 264 h and later | Training | 1979319 | 114504 | 19 | 5.78502 |
| 264 h and later | Validation | 227664 | 7241 | 18 | 3.1805643 |
| 72–144 h | Training | 2015589 | 117395 | 19 | 5.8243521 |
| 72–144 h | Validation | 232128 | 7125 | 19 | 3.0694272 |

## lag_leakage

| band | split | eligible | affected | affected_series | percent |
| --- | --- | --- | --- | --- | --- |
| 264 h and later | Training | 424390 | 0 | 0 | 0 |
| 264 h and later | Validation | 48639 | 0 | 0 | 0 |

## lag_lookup

| band | status | split | rows | series | unique_lookup_keys |
| --- | --- | --- | --- | --- | --- |
| 0–72 h | Matching value | Training | 1896927 | 31 | 638164 |
| 0–72 h | Matching value | Validation | 217707 | 31 | 217707 |
| 0–72 h | Observation key absent | Training | 117918 | 19 | 39620 |
| 0–72 h | Observation key absent | Validation | 13553 | 31 | 13553 |
| 144–264 h | Matching value | Training | 3135185 | 31 | 629821 |
| 144–264 h | Matching value | Validation | 363211 | 31 | 363211 |
| 144–264 h | Observation key absent | Training | 193440 | 19 | 38880 |
| 144–264 h | Observation key absent | Validation | 18306 | 31 | 18306 |
| 264 h and later | Matching value | Training | 1864815 | 31 | 622997 |
| 264 h and later | Matching value | Validation | 220423 | 31 | 220423 |
| 264 h and later | Observation key absent | Training | 114504 | 19 | 38264 |
| 264 h and later | Observation key absent | Validation | 7241 | 18 | 7241 |
| 72–144 h | Matching value | Training | 1898194 | 31 | 634017 |
| 72–144 h | Matching value | Validation | 225003 | 31 | 225003 |
| 72–144 h | Observation key absent | Training | 117395 | 19 | 39334 |
| 72–144 h | Observation key absent | Validation | 7125 | 19 | 7125 |

## precip_months

| split | month | rows | missing | percent |
| --- | --- | --- | --- | --- |
| Training | 2024-04 | 652860 | 652860 | 100 |
| Training | 2024-05 | 674622 | 674622 | 100 |
| Training | 2024-06 | 652860 | 652860 | 100 |
| Training | 2024-07 | 674622 | 674622 | 100 |
| Training | 2024-08 | 674622 | 674622 | 100 |
| Training | 2024-09 | 652860 | 652860 | 100 |
| Training | 2024-10 | 674622 | 674622 | 100 |
| Training | 2024-11 | 652860 | 261144 | 40 |
| Training | 2024-12 | 674622 | 0 | 0 |
| Training | 2025-01 | 674622 | 0 | 0 |
| Training | 2025-02 | 609336 | 0 | 0 |
| Training | 2025-03 | 674622 | 0 | 0 |
| Training | 2025-04 | 652860 | 0 | 0 |
| Training | 2025-05 | 674622 | 0 | 0 |
| Training | 2025-06 | 492156 | 0 | 0 |
| Validation | 2025-07 | 108810 | 0 | 0 |
| Validation | 2025-08 | 87048 | 0 | 0 |
| Validation | 2025-09 | 108810 | 0 | 0 |
| Validation | 2025-10 | 87048 | 0 | 0 |
| Validation | 2025-11 | 87048 | 0 | 0 |
| Validation | 2025-12 | 108810 | 0 | 0 |
| Validation | 2026-01 | 87048 | 0 | 0 |
| Validation | 2026-02 | 87048 | 0 | 0 |
| Validation | 2026-03 | 108810 | 0 | 0 |
| Validation | 2026-04 | 87048 | 0 | 0 |
| Validation | 2026-05 | 87048 | 0 | 0 |
| Validation | 2026-06 | 76632 | 0 | 0 |
