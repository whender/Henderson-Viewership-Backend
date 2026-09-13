"""Make the latest saved forecast canonical while retaining revision history."""
from weekly_predictions_fs import calc_error, parse_viewership


def refresh_latest_forecast(game):
    before = dict(game)
    revised = game.get('revised_predicted')
    if revised and parse_viewership(revised) is not None:
        history = list(game.get('prediction_history') or [])
        if game.get('predicted') and game['predicted'] != revised:
            history.append({key: game.get(key) for key in (
                'predicted', 'percent_error', 'accuracy', 'generated_at', 'forecast_timing'
            )})
        game['prediction_history'] = history
        game['predicted'] = game.pop('revised_predicted')
        if game.get('revision_timing'):
            game['forecast_timing'] = game['revision_timing']
        game['forecast_updated_at'] = game.get('revised_at')
    error = calc_error(game.get('predicted'), game.get('actual'))
    game['percent_error'] = error
    game['accuracy'] = ('' if error is None else '🟢🎯' if error < 5 else
                        '🟢' if error < 25 else '🟡' if error < 35 else '🔴')
    return game != before
