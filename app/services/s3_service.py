import boto3
from botocore.exceptions import ClientError
from app.config import Config
from app.logger import setup_logger
import io

logger = setup_logger('s3_service')

def upload_image_to_s3(image_buffer: io.BytesIO, filename: str) -> str:
    s3_client = boto3.client(
        's3',
        region_name=Config.S3_REGION,
        aws_access_key_id=Config.S3_ACCESS_KEY,
        aws_secret_access_key=Config.S3_SECRET_KEY
    )
    try:
        s3_client.upload_fileobj(
            image_buffer,
            Config.S3_BUCKET,
            filename,
            ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
        )
        s3_base_url = f'https://{Config.S3_BUCKET}.s3.{Config.S3_REGION}.amazonaws.com'
        img_url = f"{s3_base_url}/{filename}"
        logger.info(f"Uploaded image to S3: {img_url}")
        return img_url
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise
