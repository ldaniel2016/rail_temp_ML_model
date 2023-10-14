def calculate_hourly_values(df, param):
    h_param = f'{param}1h'
    df_hourly_param = pd.DataFrame(columns=['lat', 'lon', 'analysis_date', 'forecast_period', h_param])
    unique_latlon_pairs = df[['lat', 'lon']].drop_duplicates()

    def process_row(row_tuple):
        index, row = row_tuple
        lat, lon = row['lat'], row['lon']
        filtered_df = df[(df['lat'] == lat) & (df['lon'] == lon)]
        analysis_dates = filtered_df['analysis_date'].unique()
        result_rows = []
        for ad in analysis_dates:
            ad_filtered_df = filtered_df[filtered_df['analysis_date'] == ad]
            period = 1
            while period <= 240:  # Adjust the condition based on your specific requirement
                #print(period)
                if period < 90:
                    increment = 1
                elif period < 144:
                    increment = 3
                else:
                    increment = 6
                period_filtered_df = ad_filtered_df[ad_filtered_df['forecast_period'] == period]
                param_values = period_filtered_df[param].values
                if len(param_values) > 0:
                    param_value = param_values[0]
                    if period == 1:
                        param_value = param_value/3600 # convert to Watts (Joules/sec)
                        result_rows.append({'lat': lat, 'lon': lon, 'analysis_date': ad, 'forecast_period': period, h_param: param_value})
                    else:
                        prev_period = period - increment
                        prev_period_filtered_df = ad_filtered_df[ad_filtered_df['forecast_period'] == prev_period]
                        prev_param_values = prev_period_filtered_df[param].values
                        if len(prev_param_values) > 0:
                            prev_param_value = prev_param_values[0]
                            param_difference = param_value - prev_param_value
                            param_value = param_difference / (increment * 3600)
                            result_rows.append({'lat': lat, 'lon': lon, 'analysis_date': ad, 'forecast_period': period, h_param: param_value})
                        else: # prev_parameter has no value
                            print(param,lat, lon, period, ad)
                else: # parameter has no value
                    print(param,lat, lon, period, ad)
                period += increment  # Increment the period using the calculated increment
        return result_rows
    result_rows = [row for rows in map(process_row, unique_latlon_pairs.iterrows()) for row in rows]
    df_hourly_param = pd.DataFrame(result_rows)

    return df_hourly_param

