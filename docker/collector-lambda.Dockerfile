# Lambda image for the daily price collector.
#
# Built on AWS's official Lambda base image, which includes the Lambda runtime
# (the program that receives a trigger and calls your handler function) and
# boto3. The regular collector.Dockerfile stays for local and Compose use.

FROM public.ecr.aws/lambda/python:3.12

# Same layer-caching order as before: dependencies first, then code.
COPY requirements-collector.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-collector.txt

COPY collector.py ${LAMBDA_TASK_ROOT}/

# Not a shell command: tells the Lambda runtime which function to call
# (file collector.py, function lambda_handler).
CMD ["collector.lambda_handler"]
