-- Drop the table if it exists
DROP TABLE IF EXISTS `users`;

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS bankruptcy;

-- Switch to the created database
USE bankruptcy;

-- Create the user table
CREATE TABLE `users` (
    `name` VARCHAR(225),
    `email` VARCHAR(225),
    `password` VARCHAR(225),
    `number` VARCHAR(225)
);