// var echarts = require('echarts');

var ROOT_PATH =
    'https://cdn.jsdelivr.net/gh/apache/echarts-website@asf-site/examples';

// var chartDom = document.getElementById('main');
// var myChart = echarts.init(chartDom);
var myChart = echarts.init(document.getElementById("chart"))
var option;

// $.get(
//     '/Users/k/Downloads/image processing/bchart.json',
//     function (_rawData) {
//         run(_rawData);
//     }
// );
run(rawData)
function run(_rawData) {
    // var countries = ['Australia', 'Canada', 'China', 'Cuba', 'Finland', 'France', 'Germany', 'Iceland', 'India', 'Japan', 'North Korea', 'South Korea', 'New Zealand', 'Norway', 'Poland', 'Russia', 'Turkey', 'United Kingdom', 'United States'];
    // const countries = [
    //     'Finland',
    //     'France',
    //     'Germany',
    //     'Iceland',
    //     'Norway',
    //     'Poland',
    //     'Russia',
    //     'United Kingdom'
    // ];
    const countries = [
        'baba',
        'nvda',
        'tal',
        'tsla',
        'aapl',
        'amzn',
        'fb',
        'googl',
        'nflx',
        'se',
        'uber'
    ];

    const datasetWithFilters = [];
    const seriesList = [];
    echarts.util.each(countries, function (country) {
        var datasetId = 'dataset_' + country;
        datasetWithFilters.push({
            id: datasetId,
            fromDatasetId: 'dataset_raw',
            transform: {
                type: 'filter',
                config: {
                    and: [
                        // { dimension: 'Year', gte: 1700.01 },
                        { dimension: 'Stock', '=': country }
                    ]
                }
            }
        });
        seriesList.push({
            type: 'line',
            datasetId: datasetId,
            showSymbol: false,
            name: country,
            endLabel: {
                show: true,
                formatter: function (params) {
                    return params.value[1] + ': ' + params.value[0];
                }
            },
            labelLayout: {
                moveOverlap: 'shiftY'
            },
            emphasis: {
                focus: 'series'
            },
            encode: {
                x: 'Year',
                y: 'News_count',
                label: ['Stock', 'News_count'],
                itemName: 'Year',
                tooltip: ['News_count']
            }
        });
    });
    option = {
        animationDuration: 10000,
        dataset: [
            {
                id: 'dataset_raw',
                source: _rawData
            },
            ...datasetWithFilters
        ],
        title: {
            text: 'News number of the stock'
        },
        tooltip: {
            order: 'valueDesc',
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            nameLocation: 'middle'
        },
        yAxis: {
            name: 'News Counts'
        },
        grid: {
            right: 140
        },
        series: seriesList
    };
    myChart.setOption(option);
}