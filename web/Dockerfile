FROM php:8.1-fpm-alpine

RUN mkdir -p /var/www
WORKDIR /var/www

# Install required packages
RUN apk add --update --no-cache git

# Clone project
RUN rm -rf html
RUN git clone -b Flowbite https://github.com/TheArKaID/porcadec . --depth=1
RUN cp .env.example .env

# Install composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

# Install dependency
RUN composer install

# Setup composer
RUN php artisan key:generate
RUN php artisan migrate
RUN php artisan cache:clear && php artisan config:clear

# Expose port 9000
EXPOSE 9000

COPY . .

RUN ["chmod", "+x", "post_deploy.sh"]

CMD [ "sh", "./post_deploy.sh" ]



