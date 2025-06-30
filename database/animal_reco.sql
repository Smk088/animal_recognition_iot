-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: May 10, 2025 at 09:11 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `animal_reco`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `mobile`) VALUES
('admin', 'admin', 9894442716);

-- --------------------------------------------------------

--
-- Table structure for table `animal_detect`
--

CREATE TABLE `animal_detect` (
  `id` int(11) NOT NULL,
  `user` varchar(20) NOT NULL,
  `animal` varchar(20) NOT NULL,
  `image_name` varchar(40) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `animal_detect`
--


-- --------------------------------------------------------

--
-- Table structure for table `animal_img`
--

CREATE TABLE `animal_img` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `animal_img` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `animal_img`
--


-- --------------------------------------------------------

--
-- Table structure for table `animal_info`
--

CREATE TABLE `animal_info` (
  `id` int(11) NOT NULL,
  `animal` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `animal_info`
--

INSERT INTO `animal_info` (`id`, `animal`) VALUES
(1, 'Bear'),
(2, 'Horse'),
(3, 'Cow'),
(4, 'Elephant'),
(5, 'Goat'),
(6, 'Pig'),
(7, 'Sheep');

-- --------------------------------------------------------

--
-- Table structure for table `ani_data`
--

CREATE TABLE `ani_data` (
  `id` int(11) NOT NULL,
  `animal` varchar(30) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ani_data`
--

INSERT INTO `ani_data` (`id`, `animal`, `dtime`) VALUES
(1, 'Goat', '2021-12-08 14:44:28'),
(2, 'Elephant', '2021-12-08 14:44:34'),
(3, 'Goat', '2021-12-08 14:45:03'),
(4, 'Cow', '2021-12-08 14:45:09'),
(5, 'Goat', '2021-12-08 14:45:21'),
(6, 'Goat', '2022-03-24 16:51:15'),
(7, 'Goat', '2022-03-24 16:51:21'),
(8, 'Elephant', '2022-03-24 16:51:27'),
(9, 'Goat', '2022-03-24 16:51:33'),
(10, 'Elephant', '2022-03-24 16:51:39'),
(11, 'Elephant', '2022-03-24 16:51:45'),
(12, 'Cow', '2022-03-24 16:51:51'),
(13, 'Cow', '2022-03-24 16:51:56'),
(14, 'Horse', '2022-03-24 16:52:03'),
(15, 'Cow', '2022-03-24 16:52:09'),
(16, 'Goat', '2022-03-24 16:52:15'),
(17, 'Cow', '2022-03-24 16:52:20'),
(18, 'Horse', '2022-03-24 16:52:26'),
(19, 'Cow', '2022-03-24 16:52:32'),
(20, 'Goat', '2022-03-24 16:52:38'),
(21, 'Horse', '2022-03-25 12:40:08'),
(22, 'Elephant', '2022-03-25 12:40:14'),
(23, 'Cow', '2022-03-25 12:40:20'),
(24, 'Cow', '2022-03-25 12:40:26'),
(25, 'Goat', '2022-03-25 12:40:32'),
(26, 'Elephant', '2022-03-25 12:40:37'),
(27, 'Cow', '2022-03-25 12:40:43'),
(28, 'Cow', '2022-03-25 12:40:48'),
(29, 'Goat', '2022-03-25 12:40:54'),
(30, 'Cow', '2022-03-25 12:40:59'),
(31, 'Horse', '2022-03-25 12:46:13'),
(32, 'Elephant', '2022-03-25 12:46:18'),
(33, 'Horse', '2022-03-25 12:46:24'),
(34, 'Goat', '2022-03-25 12:46:29'),
(35, 'Elephant', '2022-03-25 12:46:35'),
(36, 'Cow', '2022-03-25 12:46:40'),
(37, 'Cow', '2022-03-25 12:46:46'),
(38, 'Horse', '2022-03-25 12:46:51'),
(39, 'Cow', '2022-03-25 12:46:57'),
(40, 'Elephant', '2022-03-25 12:47:02'),
(41, 'Cow', '2022-03-25 12:47:08'),
(42, 'Goat', '2022-03-25 12:47:13'),
(43, 'Elephant', '2022-03-25 12:47:18'),
(44, 'Goat', '2022-03-25 12:47:24'),
(45, 'Elephant', '2022-03-25 12:47:30'),
(46, 'Horse', '2022-03-25 12:47:35'),
(47, 'Elephant', '2022-03-25 12:47:41'),
(48, 'Cow', '2022-03-25 12:47:46'),
(49, 'Horse', '2022-03-25 12:47:57'),
(50, 'Goat', '2022-03-25 12:48:03'),
(51, 'Goat', '2022-03-25 12:48:09'),
(52, 'Cow', '2022-03-25 12:49:05'),
(53, 'Goat', '2022-03-25 12:49:11'),
(54, 'Goat', '2022-03-25 12:49:17'),
(55, 'Elephant', '2022-03-25 12:49:22'),
(56, 'Elephant', '2022-03-25 12:49:28'),
(57, 'Elephant', '2022-03-25 12:49:33'),
(58, 'Horse', '2022-03-25 12:49:39'),
(59, 'Horse', '2022-03-25 12:49:44'),
(60, 'Cow', '2022-03-25 12:49:50'),
(61, 'Cow', '2022-03-25 12:49:56'),
(62, 'Cow', '2022-03-25 12:50:01'),
(63, 'Cow', '2022-03-25 12:50:07'),
(64, 'Goat', '2022-03-25 12:50:12'),
(65, 'Cow', '2022-03-25 12:50:18'),
(66, 'Cow', '2022-03-25 12:50:23'),
(67, 'Elephant', '2022-03-25 12:50:29'),
(68, 'Elephant', '2022-03-25 12:50:35'),
(69, 'Horse', '2022-03-25 12:50:40'),
(70, 'Cow', '2022-03-25 12:50:46'),
(71, 'Horse', '2022-03-25 12:50:51'),
(72, 'Cow', '2022-03-25 12:50:56'),
(73, 'Horse', '2022-03-25 12:51:02'),
(74, 'Horse', '2022-03-25 12:52:24'),
(75, 'Elephant', '2022-03-25 12:52:30'),
(76, 'Horse', '2022-03-25 12:52:36'),
(77, 'Goat', '2022-03-25 12:52:41'),
(78, 'Goat', '2022-03-25 12:52:47'),
(79, 'Goat', '2022-03-25 12:52:53'),
(80, 'Cow', '2022-03-25 12:52:58'),
(81, 'Cheta', '2022-03-30 14:54:10'),
(82, 'Panda', '2022-03-30 15:09:32'),
(83, 'Lion', '2022-03-30 15:14:28'),
(84, 'Fox', '2022-03-30 15:14:34'),
(85, 'Lion', '2022-03-30 15:16:09'),
(86, 'Pig', '2022-03-30 15:16:21'),
(87, 'Monkey', '2022-03-30 15:16:39'),
(88, 'Fox', '2022-03-30 15:17:05'),
(89, 'Monkey', '2022-03-30 15:17:21'),
(90, 'ostrich', '2022-03-30 15:17:37'),
(91, 'Tiger', '2022-03-30 15:17:53'),
(92, 'Lion', '2022-03-30 15:18:09'),
(93, 'Girafee', '2022-03-30 15:18:41'),
(94, 'Monkey', '2022-03-30 15:20:02'),
(95, 'Monkey', '2022-03-30 15:21:06'),
(96, 'Tiger', '2022-03-30 16:20:02'),
(97, 'Leoprd', '2022-03-30 16:20:43'),
(98, 'Pig', '2022-03-30 16:20:58'),
(99, 'Elephant', '2022-03-30 16:21:13'),
(100, 'Fox', '2022-03-30 16:21:28'),
(101, 'Lion', '2022-03-30 16:21:43'),
(102, 'Elephant', '2022-03-30 16:22:13'),
(103, 'Lion', '2022-03-30 16:23:25'),
(104, 'Lion', '2022-03-30 16:23:53'),
(105, 'Elephant', '2022-03-30 16:24:08'),
(106, 'Panda', '2022-03-30 16:24:35'),
(107, 'Fox', '2022-03-30 16:25:02'),
(108, 'Pig', '2022-03-30 16:58:31'),
(109, 'ostrich', '2022-03-30 16:58:59'),
(110, 'Girafee', '2022-03-30 16:59:13'),
(111, 'Goat', '2022-04-24 19:57:47'),
(112, 'Elephant', '2022-04-24 21:03:56'),
(113, 'Goat', '2022-04-24 21:04:02'),
(114, 'Goat', '2023-02-21 19:27:51'),
(115, 'Horse', '2023-03-03 23:26:34'),
(116, 'Horse', '2023-03-03 23:26:40'),
(117, 'Girafee', '2023-05-07 17:33:43'),
(118, 'Leoprd', '2023-05-07 17:33:58'),
(119, 'Panda', '2025-05-05 15:16:51'),
(120, 'Leoprd', '2025-05-05 15:52:44'),
(121, 'Pig', '2025-05-05 15:52:59'),
(122, 'Bison', '2025-05-05 15:53:15'),
(123, 'Panda', '2025-05-05 16:34:00');

-- --------------------------------------------------------

--
-- Table structure for table `ani_register`
--

CREATE TABLE `ani_register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `location` varchar(50) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `ani_register`
--

INSERT INTO `ani_register` (`id`, `name`, `mobile`, `email`, `location`, `uname`, `pass`) VALUES
(1, 'Kumar', 8940228614, 'kumar@gmail.com', 'Tanjore', 'kumar', '1234'),
(2, 'sabana', 7708106571, 'sabanahussain2002@gmail.com', 'triichy', 'itrj', '123'),
(3, 'Sathish', 8942614541, 'sathish11@gmail.com', 'Karur', 'sathish', '123456'),
(4, 'Mahesh', 9958476148, 'mahesh@rndit.co.in', 'Trichy', 'mahesh', '1234'),
(5, 'Raj', 9894442716, 'raj@gmail.com', 'Chennai', 'raj', '1234');

-- --------------------------------------------------------

--
-- Table structure for table `store_data`
--

CREATE TABLE `store_data` (
  `id` int(11) NOT NULL default '0',
  `otype` varchar(20) NOT NULL,
  `name` varchar(30) NOT NULL,
  `imgname` varchar(20) NOT NULL,
  `train_img` varchar(20) NOT NULL,
  `train_st` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `store_data`
--

INSERT INTO `store_data` (`id`, `otype`, `name`, `imgname`, `train_img`, `train_st`) VALUES
(1, 'Cow', 'Cow', 'image1.jpg', '', 0),
(2, 'Cow', 'Cow', 'image2.jpg', '', 0),
(3, 'Elephant', 'Elephant', 'image2.jpg', '', 0),
(4, 'Elephant', 'Elephant', 'image4.jpg', '', 0),
(5, 'Goat', 'Goat', 'image5.jpg', '', 0),
(6, 'Goat', 'Goat', 'image6.jpg', '', 0);

-- --------------------------------------------------------

--
-- Table structure for table `train_data`
--

CREATE TABLE `train_data` (
  `id` int(11) NOT NULL,
  `animal` varchar(30) NOT NULL,
  `fimg` varchar(20) NOT NULL,
  `value1` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `train_data`
--

INSERT INTO `train_data` (`id`, `animal`, `fimg`, `value1`) VALUES
(1, 'Elephant', '1_40.jpg', 'a');
