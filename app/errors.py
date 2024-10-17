from flask import jsonify
from werkzeug.exceptions import HTTPException

def handle_exception(e):
    if isinstance(e, HTTPException):
        response = e.get_response()
        response.data = jsonify({
            "error": e.name,
            "description": e.description
        }).data
        response.content_type = "application/json"
        return response
    else:
        return jsonify({
            "error": "Internal Server Error",
            "description": str(e)
        }), 500
