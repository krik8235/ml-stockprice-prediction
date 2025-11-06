def detect_data_drift(current_data, ref_data):
    import datetime
    import json
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    from src._utils import main_logger

    if isinstance(current_data, list): current_data = pd.DataFrame(current_data)
    if ref_data is not None and isinstance(ref_data, list): ref_data = pd.DataFrame(ref_data)

    try:
        columns_to_drop = ['dt', 'timestamp_in_ms']
        current_data_clean = current_data.drop(
            columns=[col for col in columns_to_drop if col in current_data.columns],
            errors='ignore'
        )
        ref_data_clean = ref_data.drop(
            columns=[col for col in columns_to_drop if col in ref_data.columns],
            errors='ignore'
        )
        drift_report = Report(metrics=[DataDriftPreset()])
        snapshot = drift_report.run(
            reference_data=ref_data_clean,
            current_data=current_data_clean,
            timestamp=datetime.datetime.now(),
            name='data drift test'
        )
        report_json = snapshot.json()
        report_dict = json.loads(report_json)

        for item in report_dict['metrics']:
            if isinstance(item.get('value'), (float, int)):
                p_val = item['value']
                metric_id = item.get('metric_id', 'UnknownMetric')
                log_message = f'... ID: {item["id"]} Metric: {metric_id}'
                if p_val < 0.05:
                    main_logger.info(f'{log_message} P-Value: {p_val:.5f} 🚨 drift risk high')
                else:
                    main_logger.info(f'{log_message} P-Value: {p_val:.5f} ✅ no significant drift')

    except Exception as e:
        main_logger.error(f'error during detection test: {e}')
