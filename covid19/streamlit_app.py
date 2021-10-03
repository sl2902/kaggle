import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import re
import glob
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler
import datetime
from pandas.tseries.offsets import MonthBegin
from operator import attrgetter
import sparklines
import base64
from itertools import combinations
from io import BytesIO
from IPython.display import HTML
from plotly.subplots import make_subplots
from warnings import filterwarnings
import streamlit as st
from annotated_text import annotated_text

pd.set_option('MAX_ROWS', None)
pd.set_option('MAX_COLUMNS', None)
pd.set_option('display.max_colwidth', None)
pio.templates.default = "none"
filterwarnings('ignore')

path = r'/Users/home/Documents/kaggle/covid19/input/learnplatform-covid19-impact-on-digital-learning/'

# read the data
districts = pd.read_csv(os.path.join(path, 'districts_info.csv'))

# change the data type
districts['district_id'] = districts['district_id'].astype(str)

def split_range(row):
    """
    Split the range related features
    in the Districts dataset, and compute
    the midpoint
    """
    if pd.isna(row):
        return row
    matched = re.search(r'\[(.*?),\s?(.*?)\[', row)
    lb = float(matched.group(1))
    ub = float(matched.group(2))
    return (lb + ub)/2

# create new features
districts['pct_mean_black_hispanic'] = districts['pct_black/hispanic'].apply(split_range)
districts['pct_mean_free'] = districts['pct_free/reduced'].apply(split_range)
districts['county_connections_mean_ratio'] = districts['county_connections_ratio'].apply(split_range)
districts['median_pp_total_raw'] = districts['pp_total_raw'].apply(split_range)

# apply min max scaling due to the wide ranges
minmax = MinMaxScaler()
districts['scaled_black'] = minmax.fit_transform(districts['pct_mean_black_hispanic'].values.reshape(-1, 1))
districts['scaled_free'] = minmax.fit_transform(districts['pct_mean_free'].values.reshape(-1, 1))
districts['scaled_internet'] = minmax.fit_transform(districts['county_connections_mean_ratio'].values.reshape(-1, 1))
districts['scaled_investment'] = minmax.fit_transform(districts['median_pp_total_raw'].values.reshape(-1, 1))

# read the products dataset
products = pd.read_csv(os.path.join(path, 'products_info.csv'))


def create_table_bar_chart(df, var, bar_color=None, title=None):
    """
    Create a tabular bar chart
    """
    HTML(
        pd.DataFrame(df[var]\
            .value_counts(normalize=True))\
            .style\
            .format('{:.0%}')\
            .set_table_styles([{
            'selector': 'caption',
            'props': [
                ('font-size', '16px')
            ]
            }])\
          .set_caption(title)\
          .set_properties(padding='10px', border='2px solid white')\
          .bar(color=bar_color)
    )
@st.cache
def create_engagement_dataset(path, file='*.csv'):
    """
    Create engagement dataset
    """
    all_df = []
    for f in glob.glob(os.path.join(path, file)):
        df = pd.read_csv(f, parse_dates=['time'])
        df['district_id'] = f.split('/')[-1].split('.')[0]
        # df['log_engagement_index'] = np.log1p(df['engagement_index'])
        # df['scaled_access'] = minmax.fit_transform(df['pct_access'].values.reshape(-1, 1))
        # df['scaled_engagement'] = minmax.fit_transform(df['engagement_index'].values.reshape(-1, 1))
        df['usage_month'] = df['time'].dt.to_period('M')
        df['is_pandemic'] = np.where(df['time'] <= datetime.datetime.strptime('2020-07-31', '%Y-%m-%d'), 0, 1)
        all_df.append(df)
    return pd.concat(all_df, ignore_index=True)

eng_path = r'engagement_data'
daily_eng_df = create_engagement_dataset(os.path.join(path, eng_path))  

# drop records with no engagement
daily_eng_df.dropna(inplace=True)

# merge daily engagement data and products
daily_eng_df = daily_eng_df.merge(products, 
            left_on=['lp_id'], right_on=['LP ID'], how='left')

daily_eng_df['scaled_access'] = minmax.fit_transform(daily_eng_df['pct_access'].values.reshape(-1, 1))
daily_eng_df['scaled_engagement'] = minmax.fit_transform(daily_eng_df['engagement_index'].values.reshape(-1, 1))


daily_eng_df = daily_eng_df.merge(districts[['district_id', 'state', 'locale']], 
                                 left_on=['district_id'], right_on=['district_id'], how='left')

# create_table_bar_chart(daily_eng_df, 'state', bar_color=bar_color, title='Percentage of recorded engagements by State')
st.title('Share of daily engagements by State')
bar_color = '#FF7F7F'
st.dataframe(
    pd.DataFrame(daily_eng_df['state'].value_counts(normalize=True)).style\
                                                              .format('{:.0%}')\
                                                              .set_table_styles([{
                                                                'selector': 'caption',
                                                                'props': [
                                                                    ('font-size', '16px')
                                                                ]
                                                            }])\
#                                                             .set_caption('Share of daily engagements by State')\
                                                             .set_properties(padding='10px', border='2px solid white')\
                                                             .bar(color=bar_color)    
)
st.text('What do we observe')
st.text(
    '1) Utah, Connecticut and Illinois are the top 3 states that showed a \n relatively better daily engagement share compared to the rest of the States.\n'
    '2) Some of the poor performing States are: New Hampshire, Arizona, Minnesota \n and North Dakota.'
)

st.title('District-wise - Number of daily engagements')
tab1, tab2 = st.columns(2)
tab1.header('Top 5 districts')
tab1.dataframe(
    # top 5 districts in terms of number of engagements recorded
    pd.DataFrame(daily_eng_df.groupby(['state','district_id']).size())\
            .sort_values(0, ascending=False)[:5]\
            .rename(columns={0: 'count'})\
            .style\
            .set_table_styles([{
                                'selector': 'caption',
                                'props': [
                                    ('font-size', '16px')
                                ]
                            }])\
#            .set_caption('Top 5 districts - Number of daily engagements')\
            .set_properties(padding='10px', border='2px solid white')\
            .bar(color=bar_color)
)
tab2.header('Bottom 5 districts')
tab2.dataframe(
    # bottom 5 districts in terms of number of engagements recorded
    pd.DataFrame(daily_eng_df.groupby(['state','district_id']).size())\
                .sort_values(0, ascending=False)[-5:]\
                .rename(columns={0: 'count'})\
                .style\
                .set_table_styles([{
                                    'selector': 'caption',
                                    'props': [
                                        ('font-size', '16px')
                                    ]
                                }])\
#                .set_caption('Bottom 5 districts - Number of daily engagements')\
                .set_properties(padding='10px', border='2px solid white')\
                .bar(color=bar_color)
)

st.text('What do we observe')
st.text(
    '1) There is a lot of variation within a State itself. For example, district - 8784 in Illinois has the highest \n\
    registered number of daily engagements ~233K. However, in contrast, another district 5042 in the same \n\
    state has one of the lowest number of engagements - ~5K.'
)

# time-series plot of mean monthly engagement_index 
overall_mean_eng_df = daily_eng_df[['time', 'pct_access', 'engagement_index']].copy()
overall_mean_eng_df.set_index('time', inplace=True)
overall_mean_eng_df = overall_mean_eng_df.resample('1M').mean()
overall_mean_eng_df.index = overall_mean_eng_df.index - MonthBegin(1)
st.title('Mean monthly engagement index across all districts')
fig = px.line(overall_mean_eng_df, y='engagement_index',
              title='Mean monthly engagement index across all districts')
st.plotly_chart(
    fig.update_xaxes(dtick="M1",
                 tickformat="%b\n%Y"),
    use_container_width=True
)
color_grey = '#808080'
st.text('What do we observe')
annotated_text(
    ('engagement_index', '', color_grey), ' - the total page load events per 1000 \n\
    students for a given product on a given day'
)
st.text(
    '1) Schools were given orders to shut down in March. However, from this chart, it took a while before \n\
    engagement waned. The month of July witnessed the lowest engagement_index of 50 page load events per 1000 students. \n\
    However, we see a reversal in this trend, which peaks in Septemeber, and dipped slightly towards \n\
    the final quarter in that year.'
)

# time-series plot of mean monthly percent access
fig = px.line(overall_mean_eng_df, y='pct_access',
              title='Mean monthly percent of access across all districts')
st.plotly_chart(
    fig.update_xaxes(dtick="M1",
                 tickformat="%b\n%Y"),
    use_container_width=True
)

st.text('What do we observe')
annotated_text(
    ('pct_access', '', color_grey), ' - Percentage of students in the district have at least \n\
    one page-load event of a given product and on a given day'
)
st.text(
    '1) The pct_access is at its highest in January. From here onwards, it drops gradually \n\
    witnessing a steep decline starting from May onwards hitting its nadir in July. Just like \n\
    with engagement_index, there is a reversal in this trend peaking in Septemeber, and dipping \n\
    towards the last quarter of 2020.'
)

def create_data_for_various_plots(df, field, freq='1D', 
                                eng_cols=None,
                                agg_var = None, 
                                is_state_level=False,
                                is_district_level=False,
                                other_df=None,
                                ):
    """
    Create data to plot scatter plot
    This is to show daily/monthly engagement
    """
    if is_state_level:
        df = df[eng_cols].groupby(['state']).agg(agg_var).reset_index()
        return df.merge(other_df, left_on=['state'],
                     right_on=['state'], how='left')
    else:
        df = df[eng_cols].groupby(['state', 'district_id']).agg(agg_var).reset_index()
        return df.merge(other_df, left_on=['state', 'district_id'],
                     right_on=['state', 'district_id'], how='left')

# aggregate the district-wise characteristtics at the State level
states_agg = districts.groupby(['state']).agg({'pct_mean_black_hispanic': np.mean,
                                               'pct_mean_free': np.mean,
                                    'county_connections_mean_ratio': np.mean,
                                    'median_pp_total_raw': np.mean,
                                    'scaled_black': np.mean,
                                    'scaled_free': np.mean,
                                    'scaled_internet': np.mean,
                                    'scaled_investment': np.mean}).reset_index()

# prepare state level engagement data for analysis of state-wise characteristics
dist_cols = ['state', 'pct_mean_black_hispanic', 'pct_mean_free', 'county_connections_mean_ratio', 'median_pp_total_raw',
            'scaled_black', 'scaled_free', 'scaled_investment']
eng_cols = ['time', 'state', 'pct_access', 'engagement_index', 'scaled_access', 'scaled_engagement']
state_level_data_df = create_data_for_various_plots(daily_eng_df, 'time',
                                                      eng_cols = eng_cols,
                                                      agg_var = {'pct_access': np.mean,
                                                                'engagement_index': np.mean,
                                                                'scaled_access': np.mean,
                                                                'scaled_engagement': np.mean},
                                                      is_state_level=True,
                                                      other_df=states_agg[dist_cols])

district_level_data_df = create_data_for_various_plots(daily_eng_df, 'time',
                                                      eng_cols = eng_cols + ['district_id'],
                                                      agg_var = {'pct_access': np.mean,
                                                                'engagement_index': np.mean,
                                                                'scaled_access': np.mean,
                                                                'scaled_engagement': np.mean},
                                                      is_district_level=True,
                                                      other_df=districts[dist_cols + ['district_id']],
                                                      )

# aggregate the district features by state and melt the dataframe
# monthly_state_data_df = state_level_data_df.groupby('state').mean().reset_index()
state_level_data_melted_df  = pd.melt(state_level_data_df, id_vars=['state'], value_vars=['scaled_access', 'scaled_engagement', 
                                                             'scaled_black', 'scaled_free', 
                                                             'scaled_investment'], 
            value_name='feature_value', var_name='feature_parameter')
# sort the dataframe in descending order of mean value for each feature_parameter
state_level_data_melted_df = state_level_data_melted_df.groupby(['feature_parameter'])['state', \
                                                           'feature_value']\
                                                        .apply(lambda x: x.sort_values('feature_value', ascending=False))\
                                                        .reset_index()\
                                                        .drop(['level_1'], axis=1)
cols = ['scaled_access', 'scaled_engagement', 'scaled_black',
       'scaled_free', 'scaled_investment']
# add column to capture the mean value for each parameter
for col in cols:
    mean_col_value = state_level_data_melted_df.query(f'feature_parameter == "{col}"')['feature_value'].mean()
    
    subset_df = state_level_data_melted_df.query(f'feature_parameter == "{col}"')
    idx = subset_df['feature_value'].apply(lambda x: 'Above mean' if x > mean_col_value else 'Below mean').index
    state_level_data_melted_df.loc[idx, 'color'] = subset_df['feature_value'].apply(lambda x: 'Above mean' if x > mean_col_value else 'Below mean')
# rename the feature_parameter
map_labels = {
    'scaled_access': 'Percent access of at least one page load event',
    'scaled_engagement': 'Page load events per 1000 students',
    'scaled_black': 'Percent of reported Hispanic/Black students',
    'scaled_free': 'Percent of free/reduced price meal',
    'scaled_investment': 'Median per pupil expenditure'
}
state_level_data_melted_df['new_feature_parameter'] = state_level_data_melted_df['feature_parameter'].replace(map_labels)

def make_subplots_for_bar_charts(df, series,
                  colors=['rgb(255, 0, 0)', '#2ca02c'],
                  title=None
                 ):
    """
    Helper function to plot scatter charts using subplots
    """
    fig = px.bar(df, x=series, y='feature_value', facet_col='new_feature_parameter', facet_col_wrap=2, 
                facet_row_spacing=0.3, facet_col_spacing=0.1, height=1200, width=900, color='color', 
                 color_discrete_map={'Above mean': colors[1],
                                    'Below mean': colors[0]})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.layout.xaxis2.update(matches=None)
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(showlegend=False, title=title)
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:
            fig.layout[axis].title.text = ''
        if type(fig.layout[axis]) == go.layout.XAxis:
            fig.layout[axis].title.text = ''
    return fig
st.title('State-wise characteristics')
st.plotly_chart(
    make_subplots_for_bar_charts(state_level_data_melted_df, 'state',
#                             title='Bar charts displaying district features by State'
                 )
)

st.text('What do we observe')
st.text(
    '1) Utah may have the highest share of daily engagements, however, \n\
    with respect to the pct_access and engagement_index features, they are \n\
    below the mean.\n'

    '2) States such as North Dakota has a higher than average pct_access, \n\
    but the engagement_index is below the overall States mean. This is because \n\
    it has data for the first 3 months only.\n'

    '3) Arizona has the highest mean engagement_index and pct_access of all \n\
    States; it also has only one district captured in this dataset.'
    '4) Minnesota, Indiana, and Michigan, in that order, have the highest \n\
    percentage of free/reduced meal price of all States.\n'

    '5) New York, District Of Columbia, and New Jersey, in that order have \n\
    the highest median per pupil expenditure of all States.\n'

    "6) Quite a number of states haven't reported figures for percentage of \n\
    reported Black/Hispanic students, percent of free/reduced mean price and \n\
    median per pupil expenditure.\n"
)

cols_compare = ['pct_access', 'engagement_index', 'pct_mean_black_hispanic',
               'pct_mean_free', 'median_pp_total_raw', 'county_connections_mean_ratio']
dist_corr = district_level_data_df[cols_compare].corr()
N = len(dist_corr.columns)
X = dist_corr.columns.tolist()
mask = np.zeros_like(dist_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
dist_corr=dist_corr.mask(mask)
hovertext = [[f'corr({X[i]}, {X[j]}) = {dist_corr.values[i][j]:.0%}' if i > j else '' for j in range(N)] for i in range(N)]
heat = go.Heatmap(z=dist_corr,
                  x=X,
                  y=X,
                  xgap=1, ygap=1,
                  colorscale='rdylgn',
                  colorbar_thickness=20,
                  colorbar_ticklen=3,
                  hovertext=hovertext,
                  hoverinfo='text'
                   )


title = 'Correlation plot for district level features'               

layout = go.Layout(title_text=title, title_x=0.5, 
                   width=800, height=800,
                   xaxis_showgrid=False,
                   yaxis_showgrid=False,
                   yaxis_automargin=True,
                   xaxis_automargin=True,
                   yaxis_autorange='reversed')
   
fig=go.Figure(data=[heat], layout=layout)        
st.plotly_chart(
    fig,
    use_container_width=True
)

st.text('What do we observe')
st.text(
    '1) Not all district features are strong correlated.\n'

    '2) The only relationships that stand out are: pct_access \n\
    versus engagement_index. b) engagement_index versus \n\
    median_pp_total_raw. c) pct_mean_free versus pct_mean_black_hispanic. \n\
    All of these relationships are positively correlated.\n'

    '3) There is a moderately strong positive correlation \n\
    between pct_access and county_connections_means_ratio.'
)
st.cache()
def create_parallel_coord(df, states):

    fig = go.Figure(data=
                    go.Parcoords(
                    line = dict(color=df['district_id'],
                              colorscale = [[0,'purple'],[0.5,'lightseagreen'],[1,'gold']]),
                    dimensions = list([
                        dict(
                                label = 'Districts', values = df['district_id']),
                        dict(range = [0, 1],
                                label = 'Black/Hispanic', values = df['scaled_black']),
                        dict(range = [0, 1],
                                label = 'Discounted meal', values = df['scaled_free']),
#                         dict(range = [0, .2],
#                                 label = 'Internet', values = df['scaled_internet']),
                        dict(range = [0, 1],
                                constraintrange = [0, .5], 
                                label = 'Investment', values = df['scaled_investment']),
                        dict(range = [0, .2],
                                label = 'Access', values = df['scaled_access']),
                        dict(range = [0, .007],
                                label = 'Engagement', values = df['scaled_engagement']),

                    ])
        )
    )


    fig.update_layout(title=f'Relationship between the district features in the state of {state}')
    return fig

state = 'New York'
district_level_data_df['district_id'] = district_level_data_df['district_id'].astype('int32')
st.title('District-wise characteristics for NY')
st.plotly_chart(
    create_parallel_coord(district_level_data_df.query(f'state == "{state}"'), state),
    use_container_width=True
)

@st.cache()
def data_for_sparkline(df, grp_var, time_var, max_time='2020-12',
                              min_time='2020-01',
                              agg_var=None,
                              is_state=False,
                              is_district=False,
                              is_product=False, 
                              is_product_eng=False,
                              is_product_combo=False,
                              is_product_combo_eng=False,
                              is_multi_level=False, 
                              cust_sparkline=True):
    """
    Create data for sparkline
    """
    tmp_df = df.copy()
    if is_state:
        g = tmp_df.groupby(grp_var).agg({agg_var: np.sum,})
        g = g.groupby(level=[0]).apply(lambda x: x/x.sum()).reset_index()
    elif is_district:
        g = tmp_df.groupby(grp_var).agg({agg_var: np.sum,})
        g = g.groupby(level=[0, 1]).apply(lambda x: x/x.sum()).reset_index()
    elif is_product:
        g = tmp_df.groupby(grp_var).size()
        if is_multi_level:
            g = g.groupby(level=[0, 1]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
        else:
            g = g.groupby(level=[0]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
    elif is_product_eng:
        g = tmp_df.groupby(grp_var).agg({agg_var: np.sum,})
        if is_multi_level:
            g = g.groupby(level=[0, 1]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
        else:
            g = g.groupby(level=[0]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
    elif is_product_combo:
        g = tmp_df.groupby(grp_var).size()
        if is_multi_level:
            g = g.groupby(level=[0, 1, 2]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
        else:
            g = g.groupby(level=[0, 1]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
    elif is_product_combo_eng:
        g = tmp_df.groupby(grp_var).agg({agg_var: np.sum,})
        if is_multi_level:
            g = g.groupby(level=[0, 1, 2]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
        else:
            g = g.groupby(level=[0, 1]).apply(lambda x: x/x.sum()).reset_index().rename(columns={0: 'usage_value'})
    
    # grp_mean = g.iloc[:, -1].mean() # compute mean for the last column
    g['usage_month'] = g['usage_month'].astype(str)
    g = g.pivot_table(index=grp_var[:-1], columns='usage_month', fill_value=0)
    g.columns = g.columns.droplevel() # drop usage_value
    g = g.rename_axis(None, axis=1) # remove usage_month
    if cust_sparkline:
        g['trend'] = g.apply(custom_sparkline, axis=1)
    else:
        g['trend'] = g.apply(lambda x: sparklines.sparklines(x)[0], axis=1)
    g['growth'] = np.round((g[max_time] / g[min_time]) ** (1/12) - 1, 2)
    g['growth'] = g['growth'].replace(np.inf, 0).replace(np.nan, 0)
    return g

def highlight_table(row, threshold=.05):
    """
    Helper function to highlight cells
    in a Pandas dataframe
    """
    if isinstance(row[0], str): return
    return [
        'background-color: #FF7F7F; color: white' if cell <= threshold
        else 'background-color: green; color: white'
        for cell in row
    ]

@st.cache()
def custom_sparkline(data, figsize=(3, 0.25), **kwags):
    """
    Create a sparkline chart
    https://github.com/iiSeymour/sparkline-nb/blob/master/sparkline-nb.ipynb
    """
    data = list(data)
    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
    ax.plot(data)
    for k, v in ax.spines.items():
        v.set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.plot(len(data)-1, data[len(data)-1], 'r.', linewidth=2)
    
    # ax.fill_between(range(len(data)), data, len(data)*[min(data)], alpha=0.1)
    
    img = BytesIO()
    plt.savefig(img, transparent=True, dpi=100, bbox_inches='tight')
    # img.seek(0)
    plt.close()
    
    return f'<img src="data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"/>'
    # return display(HTML(df.to_html(escape=False))


# subset the daily engagement data to analyze the month on month
# mean engagement at state level
tmp_df = daily_eng_df[['state', 'district_id', 'usage_month', 'scaled_engagement', 'scaled_access']].copy()
state_wise_mom_engagement_growth = data_for_sparkline(tmp_df, ['state', 'usage_month'], 'time', max_time='2020-12',
                              min_time='2020-01',
                              agg_var='scaled_engagement',
                              is_state=True,
                              cust_sparkline=False)

# mean_eng_threshold = daily_eng_df.groupby(['state'.mean()
grad_cols = state_wise_mom_engagement_growth.columns.drop(['trend', 'growth']).tolist()
st.title('Share in engagement index by State')
st.dataframe(
    state_wise_mom_engagement_growth.sort_values(['growth', 'state'], ascending=[False, True], kind='mergesort').style\
                                            .format('{:.1%}', subset=['growth'])\
                                            .format('{:.1%}', subset=state_wise_mom_engagement_growth.columns.drop(['trend', 'growth']))\
                                            .set_table_styles([{
                                                'selector': 'caption',
                                                'props': [
                                                    ('font-size', '16px')
                                                ]
                                            }])\
#                                            .set_caption('Share in engagement index by State')\
                                            .set_properties(padding='10px', border='2px solid white')\
                                            .background_gradient(cmap='RdYlGn', subset=grad_cols, axis=1)\
                                            .background_gradient(cmap='RdYlGn', subset=['growth'], axis=0)
  #                                          .apply(highlight_table, args=(state_eng_mean, ), axis=0)\
)

# subset the daily engagement data to analyze the month on month
# mean engagement at state level
district_wise_mom_engagement_growth = data_for_sparkline(tmp_df, ['state', 'district_id', 'usage_month'], 'time', max_time='2020-12',
                              min_time='2020-01',
                              agg_var='scaled_engagement',
                              is_district=True,
                              cust_sparkline=False)


# mean_eng_threshold = daily_eng_df['scaled_engagement'].mean()
st.title('Share in engagement index of top 10 Districts')
st.dataframe(
    district_wise_mom_engagement_growth.sort_values(['growth', 'state'], ascending=[False, True], kind='mergesort')[:10].style\
                                            .format('{:.1%}', subset=['growth'])\
                                            .format('{:.1%}', subset=district_wise_mom_engagement_growth.columns.drop(['trend', 'growth']))\
                                            .set_table_styles([{
                                                'selector': 'caption',
                                                'props': [
                                                    ('font-size', '16px')
                                                ]
                                            }])\
 #                                           .set_caption('Share in engagement index of top 10 Districts')\
                                            .set_properties(padding='10px', border='2px solid white')\
                                            .background_gradient(cmap='RdYlGn', subset=grad_cols, axis=1)\
                                            .background_gradient(cmap='RdYlGn', subset=['growth'], axis=0)
 #                                           .apply(highlight_table, args=(mean_dist_eng, ), axis=0)
)

st.title('Share in engagement index of bottom 10 Districts')
st.dataframe(
    district_wise_mom_engagement_growth.sort_values(['growth', 'state'], ascending=[False, True], kind='mergesort')[-10:].style\
                                            .format('{:.1%}', subset=['growth'])\
                                            .format('{:.1%}', subset=district_wise_mom_engagement_growth.columns.drop(['trend', 'growth']))\
                                            .set_table_styles([{
                                                'selector': 'caption',
                                                'props': [
                                                    ('font-size', '16px')
                                                ]
                                            }])\
                                            .set_caption('Share in engagement index of bottom 10 Districts')\
                                            .set_properties(padding='10px', border='2px solid white')\
                                            .background_gradient(cmap='RdYlGn', subset=grad_cols, axis=1)\
                                            .background_gradient(cmap='RdYlGn', subset=['growth'], axis=0)
 #                                           .apply(highlight_table, args=(mean_dist_eng, ), axis=0)
) 

# Product usage
# filter the null values in the LP ID and state level
lp_daily_eng_df = daily_eng_df[(daily_eng_df['LP ID'].notnull()) & (daily_eng_df['state'].notnull())].reset_index(drop=True)

monthly_product_usage = lp_daily_eng_df.groupby(['time'])['LP ID'].size()
#monthly_product_usage.set_index('time', inplace=True)
monthly_product_usage = monthly_product_usage.resample('1M').sum() / monthly_product_usage.sum()
monthly_product_usage.index = monthly_product_usage.index - MonthBegin(1)

# plot the monthly share of product usage
fig = px.line(monthly_product_usage, y='LP ID', title='Monthly frequency of product usage', 
       )
max_yaxis = monthly_product_usage.max()
fig.update_yaxes(tickformat=".0%",
                range=[0, np.round(max_yaxis, 2)],
                )
st.plotly_chart(
    fig.update_xaxes(dtick="M1",
                 tickformat="%b\n%Y")
)
st.title('Product characteristics')
tab1, tab2 = st.columns(2)
tab1.header('Share of top 10 daily used products')
tab1.dataframe(
        pd.DataFrame(lp_daily_eng_df['Product Name']\
                .value_counts(normalize=True))[:10]\
                .style\
                .format('{:.0%}')\
                .set_table_styles([{
                    'selector': 'caption',
                    'props': [
                        ('font-size', '16px')
                    ]
                }])\
    #            .set_caption('Share of top 10 daily used products')\
                .set_properties(padding='10px', border='2px solid white')\
                .bar(color=bar_color)
    )
tab2.header('Share of Sector(s)')
tab2.dataframe(
    pd.DataFrame(lp_daily_eng_df['Sector(s)']\
            .value_counts(normalize=True))[-10:]\
            .style\
            .format('{:.0%}')\
            .set_table_styles([{
                'selector': 'caption',
                'props': [
                    ('font-size', '16px')
                ]
             }])\
#            .set_caption('Share of Sector(s)')\
            .set_properties(padding='10px', border='2px solid white')\
            .bar(color=bar_color)
)
st.title('Share of top 10 Provider/Company Name')
st.dataframe(
    pd.DataFrame(lp_daily_eng_df['Provider/Company Name']\
                .value_counts(normalize=True))[:10]\
                .style\
                .format('{:.0%}')\
                .set_table_styles([{
                    'selector': 'caption',
                    'props': [
                        ('font-size', '16px')
                    ]
                }])\
    #            .set_caption('Share of top 10 Provider/Company Name')\
                .set_properties(padding='10px', border='2px solid white')\
                .bar(color=bar_color)
)
#tab3, tab4 = st.columns(2)
st.title('Share of top 10 Primary Essential Function')
st.dataframe(
    pd.DataFrame(lp_daily_eng_df['Primary Essential Function']\
            .value_counts(normalize=True))[:10]\
            .style\
            .format('{:.0%}')\
            .set_table_styles([{
                'selector': 'caption',
                'props': [
                    ('font-size', '16px')
                ]
             }])\
#            .set_caption('Share of top 10 Primary Essential Function')\
            .set_properties(padding='10px', border='2px solid white')\
            .bar(color=bar_color)
)

st.stop()
