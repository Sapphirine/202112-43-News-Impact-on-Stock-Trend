const upColor = '#ec0000';
const upBorderColor = '#8A0000';
const downColor = '#00da3c';
const downBorderColor = '#008F28';

function splitData(rawData) {
    const categoryData = [];
    const values = [];
    for (var i = 0; i < rawData.length; i++) {
        categoryData.push(rawData[i].splice(0, 1)[0]);
        values.push(rawData[i]);
    }
    return {
        categoryData: categoryData,
        values: values
    };
}
function calculateMA(dayCount, data0) {
    var result = [];
    for (var i = 0, len = data0.values.length; i < len; i++) {
        if (i < dayCount) {
            result.push('-');
            continue;
        }
        var sum = 0;
        for (var j = 0; j < dayCount; j++) {
            sum += +data0.values[i - j][1];
        }
        result.push(sum / dayCount);
    }
    return result;
}
function buildCandlestickChart(Stock_Data, Stock_News, Stock_Images)
{
    var data0 = splitData(Stock_Data)
    var myChart = echarts.init(document.getElementById("chart1"))
    var option = {
        // title: {
        //     text: 'candlestick',
        //     left: 0
        // },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        legend: {
            data: ['daily K', 'MA5', 'MA10', 'MA20', 'MA30']
        },
        grid: {
            left: '10%',
            right: '10%',
            bottom: '15%'
        },
        xAxis: {
            type: 'category',
            data: data0.categoryData,
            scale: true,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false },
            min: 'dataMin',
            max: 'dataMax'
        },
        yAxis: {
            scale: true,
            splitArea: {
                show: true
            }
        },
        dataZoom: [
            {
                type: 'inside',
                start: 50,
                end: 100
            },
            {
                show: true,
                type: 'slider',
                top: '90%',
                start: 50,
                end: 100
            }
        ],
        series: [
            {
                name: 'daily K',
                type: 'candlestick',
                data: data0.values,
                itemStyle: {
                    color: upColor,
                    color0: downColor,
                    borderColor: upBorderColor,
                    borderColor0: downBorderColor
                },
                markPoint: {
                    label: {
                        formatter: function (param) {
                            return param != null ? Math.round(param.value) + '' : '';
                        }
                    },
                    data: [
                        {
                            name: 'Mark',
                            coord: ['2013/5/31', 2300],
                            value: 2300,
                            itemStyle: {
                                color: 'rgb(41,60,85)'
                            }
                        },
                        {
                            name: 'highest value',
                            type: 'max',
                            valueDim: 'highest'
                        },
                        {
                            name: 'lowest value',
                            type: 'min',
                            valueDim: 'lowest'
                        },
                        {
                            name: 'average value on close',
                            type: 'average',
                            valueDim: 'close'
                        }
                    ],
                    tooltip: {
                        formatter: function (param) {
                            return param.name + '<br>' + (param.data.coord || '');
                        }
                    }
                },
                markLine: {
                    symbol: ['none', 'none'],
                    data: [
                        [
                            {
                                name: 'from lowest to highest',
                                type: 'min',
                                valueDim: 'lowest',
                                symbol: 'circle',
                                symbolSize: 10,
                                label: {
                                    show: false
                                },
                                emphasis: {
                                    label: {
                                        show: false
                                    }
                                }
                            },
                            {
                                type: 'max',
                                valueDim: 'highest',
                                symbol: 'circle',
                                symbolSize: 10,
                                label: {
                                    show: false
                                },
                                emphasis: {
                                    label: {
                                        show: false
                                    }
                                }
                            }
                        ],
                        {
                            name: 'min line on close',
                            type: 'min',
                            valueDim: 'close'
                        },
                        {
                            name: 'max line on close',
                            type: 'max',
                            valueDim: 'close'
                        }
                    ]
                }
            },
            {
                name: 'MA5',
                type: 'line',
                data: calculateMA(5, data0),
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            },
            {
                name: 'MA10',
                type: 'line',
                data: calculateMA(10, data0),
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            },
            {
                name: 'MA20',
                type: 'line',
                data: calculateMA(20, data0),
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            },
            {
                name: 'MA30',
                type: 'line',
                data: calculateMA(30, data0),
                smooth: true,
                lineStyle: {
                    opacity: 0.5
                }
            }
        ]
    };
    var news_data = []
    myChart.setOption(option)
    myChart.on('click', function (params) {
        var picBox = document.getElementById("chart2")
        picBox.style.display = "none";
        daily_images = Stock_Images[params.name]
        console.log(daily_images)
        changePics(daily_images)
        news_data = Stock_News[params.name]
        buildNewsChart(news_data);})
}
function changePics(daily_images) {
    var picBox = document.getElementById("chart2")
    picBox.style.display = "none";
    // if((typeof daily_images != "undefined")){
    if(daily_images.length == 4){
        picBox.style.display = "";
        var pic, detail;
        for (var j = 0; j < 4; j++){
            pic = document.getElementById("pic"+(j+1));
            detail = document.getElementById("detail"+(j+1));
            $(detail).empty();
            news_info = daily_images[j]
            var inner ="";
            var p_html = "<p>" + news_info['title']+ "</p>";
            var a_html = "<a href=" + news_info['url'] +">" + "enter</a>"
            inner += p_html + a_html
            $(detail).append(inner);
            var img_src = '/static/images/' + news_info['image']
            img_src = news_info['image']
            console.log(img_src)
            pic.width = 240
            pic.height = 120
            pic.src = img_src;
        }
    }
}
