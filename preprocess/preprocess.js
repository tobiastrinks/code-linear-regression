const csv = require('csv-parser');
const fs = require('fs');
const results = [];

function parseRawCsv() {
    return new Promise((resolve => {
        fs.createReadStream('../kaggle/data.csv')
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            resolve(results)
        });
    }));
}

function writeToOutputCsv(houses) {
    const preprocessedFeaturesIS = fs.createWriteStream('preprocessed_features.csv');
    const preprocessedNormalizedFeaturesIS = fs.createWriteStream('preprocessed_normalized_features.csv');
    const priceIS = fs.createWriteStream('prices.csv');

    const preprocessedFeaturesCsvColumns = [
        'date',
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'view',
        'condition',
        'sqft_above',
        'sqft_basement',
        'yr_built',
        'yr_renovated',
        'city_mean_price'
        // 'street', -> is unique and therefore not useful
        // 'city', -> is covered in statezip
        // 'statezip', -> converted to average price of city -> city_mean_price
        // 'country' -> US only and therefore not useful
    ];
    const preprocessedNormalizedFeaturesCsvColumns = [
        'waterfront'
    ];

    houses.forEach(house => {
        preprocessedFeaturesIS.write(
            preprocessedFeaturesCsvColumns.map(key => house[key]).join(',') + '\n'
        );
        preprocessedNormalizedFeaturesIS.write(
            preprocessedNormalizedFeaturesCsvColumns.map(key => house[key]).join(',') + '\n'
        );
        priceIS.write(
            house.price + '\n'
        );
    });
    preprocessedFeaturesIS.close();
    preprocessedNormalizedFeaturesIS.close();
    priceIS.close();
}

async function main() {
    const houses = shuffle(await parseRawCsv());

    const housesGroupedByStateZip = groupBy(houses, house => house.statezip);

    const normalizedHouses = houses.map(house => {
        house.date = new Date(house.date).getTime();
        house.waterfront = parseInt(house.waterfront);

        const sameZipCityPrices = housesGroupedByStateZip[house.statezip].map(house => parseFloat(house.price));
        house.city_mean_price = Math.ceil(sameZipCityPrices.reduce((acc, cur) => acc + cur, 0) / sameZipCityPrices.length);

        return house;
    });

    writeToOutputCsv(normalizedHouses);
}

main();


function groupBy(list, keyGetter) {
    const map = {};
    list.forEach((item) => {
         const key = keyGetter(item);
         const collection = map[key];
         if (!collection) {
             map[key] = [item];
         } else {
             collection.push(item);
         }
    });
    return map;
}

function shuffle(list) {
    for (let i=0; i<list.length; i++) {
        const newPos = Math.ceil(Math.random() * list.length - 1);
        const cache = list[i];
        list[i] = list[newPos];
        list[newPos] = cache;
    }
    return list;
}
