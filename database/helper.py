from src.User.model import UserModel


def get_users(request):
    limit = request.args.get('limit', None)
    order_by = request.args.get('order_by', None)
    filter_obj = {}

    for key in request.args:
        if key not in ['limit', 'order_by']:
            filter_obj[key] = request.args[key]

    users = UserModel.query \
                .filter_by(**filter_obj) \
                .order_by(order_by) \
                .limit(limit) \
                .all()

    return users

