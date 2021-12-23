function buildNewsChart(news_data)
{
    var chart0 = document.getElementById("chart0");
    chart0.style.display = ""
    var label1 = ['positive','negative','neutral']
    var label2 = ["policy","industry", "market", "investment", "business","else"]

    var label = label1.concat(label2)
    var inner_data = [
        {value:0, name:'positive'},
        {value:0, name:'negative'},
        {value:0, name:'neutral'}
    ]
    // for (var j=0; j < inner_data.length; j++)
    //     inner_data[j].value = news_data[j].length
    for (var j=0; j < inner_data.length; j++)
        inner_data[j].value = news_data['cnt'][j]
    var dataA=[];
    dataA = news_data.topics
    var option = {
        tooltip : {
            trigger: 'item',
            formatter: "{a} <br/>{b} : {c} ({d}%)"
        },
        legend: {
            orient : 'vertical',
            x : 'left',
            data: label
        },
        toolbox: {
            show : true,
            feature : {
                mark : {show: true},
                dataView : {show: true, readOnly: false},
                magicType : {
                    show: true,
                    type: ['pie', 'funnel']
                },
                restore : {show: true},
                saveAsImage : {show: true}
            }
        },
        calculable : false,
        series : [
            {
                name:'news number',
                type:'pie',
                selectedMode: 'single',
                radius : [0, 35],

                // for funnel
                x: '20%',
                width: '40%',
                funnelAlign: 'right',
                max: 1548,

                itemStyle : {
                    normal : {
                        label : {
                            position : 'inner'
                        },
                        labelLine : {
                            show : false
                        }
                    }
                },
                data:inner_data
            },
            {
                name:'impact',
                type:'pie',
                radius : [50, 70],

                // for funnel
                x: '20%',
                width: '40%',
                funnelAlign: 'left',
                max: 1048,
                labelLine: {
                    length: 30
                },
                label: {
                    formatter: '{a|{a}}{abg|}\n{hr|}\n  {b|{b}：}{c}  {per|{d}%}  ',
                    backgroundColor: '#F6F8FC',
                    borderColor: '#8C8D8E',
                    borderWidth: 1,
                    borderRadius: 4,
                    rich: {
                        a: {
                            color: '#6E7079',
                            lineHeight: 22,
                            align: 'center'
                        },
                        hr: {
                            borderColor: '#8C8D8E',
                            width: '100%',
                            borderWidth: 1,
                            height: 0
                        },
                        b: {
                            color: '#4C5058',
                            fontSize: 10,
                            fontWeight: 'bold',
                            lineHeight: 33
                        },
                        per: {
                            color: '#fff',
                            backgroundColor: '#4C5058',
                            padding: [3, 4],
                            borderRadius: 4
                        }
                    }
                },
                data:dataA
                /*[
                   {value:335, name:'直达'},
                    {value:310, name:'邮件营销'},
                    {value:234, name:'联盟广告'},
                    {value:135, name:'视频广告'},
                    {value:1048, name:'百度'},
                    {value:251, name:'谷歌'},
                    {value:147, name:'必应'},
                    {value:102, name:'其余'}
                ]*/
            }
        ]
    };

    var myChart;

    myChart = echarts.init(document.getElementById('chart0'));
    // 使用刚指定的配置项和数据显示图表。

    myChart.setOption(option);
    myChart.on('click', function (params) {
        if(params.seriesIndex==0){
            for(var i=0;i<option.series[0].data.length;i++){
                option.series[0].data[i].selected=false;
            }
            var selected=params.data;
            selected.selected=true;
            console.log(selected);
            console.log( option.series[0].data[params.dataIndex]);
            option.series[0].data[params.dataIndex]=selected;
            //  option.series[1].data=dataA;
            option.series[1].data=news_data.topics[params.dataIndex];

            console.log(option);
            myChart.clear();
            myChart.setOption(option);
        }
        else{
            window.open(params.data.url)
        }
    });
}