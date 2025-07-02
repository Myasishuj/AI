import pandas as pd
import unicodedata
import geonamescache
from rapidfuzz import process, fuzz
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ——— Helpers —————————————————————————

def normalize(s: str) -> str:
    """Strip accents, lowercase, trim."""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )

# ——— 1. Build offline lookup tables —————————————————

# Load geonamescache data
gc = geonamescache.GeonamesCache()
cities = gc.get_cities()        # dict(id → city info)
countries = gc.get_countries()  # dict(ISO2 → country info)

# Build DataFrame of normalized city names + country ISO2 + coords
rows = []
for city_info in cities.values():
    city_n = normalize(city_info["name"])
    iso2   = city_info["countrycode"]
    lat    = float(city_info["latitude"])
    lon    = float(city_info["longitude"])
    rows.append((city_n, iso2, lat, lon))

cities_df = pd.DataFrame(rows, columns=["city_n","iso2","latitude","longitude"])

# Map normalized country name → ISO2 code
country_name_to_iso = {
    normalize(info["name"]): iso2
    for iso2, info in countries.items()
}

# ——— 2. Load your dataset & normalize —————————————————

INPUT_CSV  = r"C:\School\AI\SocialMediaUsersDataset.csv"
OUTPUT_CSV = "SocialMediaUsersDataset_with_all_coordinates.csv"

df = pd.read_csv(INPUT_CSV)
df["city_n"]    = df["City"].apply(normalize)
df["country_n"] = df["Country"].apply(normalize)
df["iso2"]      = df["country_n"].map(country_name_to_iso)

# ——— 3. Exact offline merge ——————————————————————

merged = df.merge(
    cities_df,
    how="left",
    left_on=["city_n","iso2"],
    right_on=["city_n","iso2"],
)

# Count how many got matched
matched    = merged["latitude"].notna().sum()
total_rows = len(merged)
print(f"🔍 Exact matches via geonamescache: {matched}/{total_rows}")

# ——— 4. Fuzzy-match the rest ————————————————————

# Identify unique city-iso2 combos still missing
to_fix = (
    merged[merged["latitude"].isna()]
    .loc[:, ["city_n","iso2"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"🤖 Fuzzy/API needed for {len(to_fix)} unique combos")

# Setup fuzzy & geocoder
geolocator = Nominatim(user_agent="fast_geocoder")
geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

cache = {}

def lookup_combo(city_n, iso2):
    key = (city_n, iso2)
    if key in cache:
        return cache[key]

    # 1) Fuzzy match within the same country
    subset = cities_df[cities_df["iso2"] == iso2]
    if not subset.empty:
        match, score, idx = process.extractOne(
            city_n, subset["city_n"], scorer=fuzz.partial_ratio
        )
        if score >= 90:
            lat = subset.at[idx, "latitude"]
            lon = subset.at[idx, "longitude"]
            cache[key] = (lat, lon)
            return cache[key]

    # 2) Fallback to online geocoding
    try:
        loc = geocode(f"{city_n}, {countries[iso2]['name']}")
        if loc:
            cache[key] = (loc.latitude, loc.longitude)
            return cache[key]
    except Exception:
        pass

    cache[key] = (None, None)
    return cache[key]

# Apply fixes back into merged
for idx, (city_n, iso2) in to_fix.itertuples(index=False):
    lookup_combo(city_n, iso2)

def fill_coords(row):
    if pd.notna(row["latitude"]):
        return row["latitude"], row["longitude"]
    return cache.get((row["city_n"], row["iso2"]), (None, None))

merged[["latitude","longitude"]] = merged.apply(
    fill_coords, axis=1, result_type="expand"
)

# ——— 5. Save final CSV —————————————————————

# Drop helper cols
final = merged.drop(columns=["city_n","country_n","iso2"])
final.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Done! Saved to '{OUTPUT_CSV}'")
